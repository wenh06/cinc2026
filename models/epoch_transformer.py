"""Epoch-sequence Transformer for CinC 2026.

Each PSG night is represented as a variable-length sequence of N × 30-second
epochs (typically 730–1100 per night). Each epoch is encoded as a
``CAISR_EPOCH_DIM``-dimensional feature vector derived from the CAISR
algorithmic annotations (see ``dataset.build_epoch_features``).

A Transformer encoder processes the full-night sequence and produces a single
binary logit for the cognitive-impairment prediction.

Architecture
------------
1. Linear projection + LayerNorm: (B, T, caisr_dim) → (B, T, d_model)
2. Fixed sinusoidal positional encoding (added, not learned; handles variable T)
3. Transformer encoder with Pre-LayerNorm (``norm_first=True``) for stability
4. Masked mean pooling over non-padding epochs → (B, d_model)
5. FiLM demographic modulation (``DemographicEncoder`` with ``mode="film"``)
6. MLP classification head → scalar logit per sample → BCEWithLogitsLoss
"""

import math
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from outputs import CINC2026Outputs

from .building_blocks import DemographicEncoder

__all__ = ["EpochTransformer"]


def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Return a ``(max_len, d_model)`` sinusoidal positional encoding matrix."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EpochTransformer(nn.Module, SizeMixin, CkptMixin):
    """Epoch-sequence Transformer for CinC 2026.

    Parameters
    ----------
    config : CFG, optional
        Model configuration. Defaults to ``ModelCfg.epoch_transformer``.
    **kwargs
        Any key overrides the corresponding field in ``config``.

    Input dict keys (``input_tensors``)
    ------------------------------------
    epoch_features : Tensor, shape ``(B, T, caisr_feat_dim)``
        CAISR epoch feature vectors (output of ``dataset.build_epoch_features``).
    demographics : Tensor, shape ``(B, demographic_dim)``
        Normalised demographic features ``[age/100, sex (0=F/1=M), bmi/50]``.
        Missing/invalid values default to ``[0.6, 0.0, 0.5]`` (≈ 60 yr, Female, 25 BMI).
    padding_mask : BoolTensor, shape ``(B, T)``, optional
        ``True`` marks padding positions (matches PyTorch
        ``TransformerEncoder.src_key_padding_mask`` convention).
    labels : Tensor, shape ``(B,)``, optional
        Binary ground-truth labels (float, 0 or 1) used for loss computation.

    Output dict keys
    ----------------
    ci_logits : Tensor, shape ``(B, 1)``
        Raw BCEWithLogitsLoss-compatible logit.  Shaped (B, 1) for
        ``CINC2026Outputs`` compatibility.
    ci_prob : Tensor, shape ``(B, 2)``
        ``[P(CI=0), P(CI=1)]`` derived via sigmoid.  ``ci_prob[:, 1]`` is the
        positive-class probability used by ``CINC2026Outputs`` for thresholding.
    cognitive_impairment : LongTensor, shape ``(B,)``
        Hard prediction at threshold 0.5.
    ci_loss : Tensor or None
        Scalar ``BCEWithLogitsLoss`` value when ``labels`` are provided.
    """

    __name__ = "EpochTransformer"

    # Handles up to ~17-hour recordings at 30 s/epoch (2048 × 30 s ≈ 17 h)
    _MAX_PE_LEN: int = 2048

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            self.config = deepcopy(ModelCfg.epoch_transformer)
        else:
            self.config = deepcopy(config)
        self.config.update(kwargs)

        self.classes = self.config.classes
        self.n_classes = len(self.classes)

        caisr_dim = self.config.caisr_feat_dim  # 21
        d_model = self.config.d_model  # 128
        nhead = self.config.nhead  # 4
        num_layers = self.config.num_layers  # 4
        dim_ff = self.config.dim_feedforward  # 512
        dropout = self.config.dropout  # 0.1
        activation = self.config.activation  # "gelu"

        # ── 1. Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(caisr_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. Fixed sinusoidal positional encoding ───────────────────────────
        self.register_buffer("pos_encoding", _build_sinusoidal_pe(self._MAX_PE_LEN, d_model))

        # ── 3. Transformer encoder (Pre-LN for numerical stability) ──────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm: more stable than Post-LN
        )
        # Final layer norm after the stack (standard for Pre-LN Transformers)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoids a warning when mask is provided
        )

        # ── 4. Demographic modulation (FiLM) ─────────────────────────────────
        dem_cfg = self.config.dem_encoder
        if dem_cfg.enable:
            self.dem_encoder = DemographicEncoder(
                dem_input_dim=dem_cfg.input_dim,
                feature_dim=d_model,
                mode=dem_cfg.mode,
                hidden_dim=dem_cfg.hidden_dim,
            )
        else:
            self.dem_encoder = None

        # ── 5. Classification head ────────────────────────────────────────────
        self.clf = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.criterion = setup_criterion(self.config.criterion, **self.config.get("criterion_kw", {}))

    # ─── Forward pass ────────────────────────────────────────────────────────

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, None]]:
        epoch_features = input_tensors["epoch_features"].to(self.device).to(self.dtype)
        demographics = input_tensors["demographics"].to(self.device).to(self.dtype)

        padding_mask = input_tensors.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)  # (B, T) bool, True=padding

        B, T, _ = epoch_features.shape

        # 1. Project + sinusoidal positional encoding
        x = self.input_proj(epoch_features)  # (B, T, d_model)
        x = x + self.pos_encoding[:T].unsqueeze(0)  # broadcast over batch dim

        # 2. Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # 3. Masked mean pooling — average only non-padding epochs
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)  # (B, d_model)

        # 4. FiLM demographic modulation
        if self.dem_encoder is not None:
            scale, shift = self.dem_encoder(demographics)
            pooled = self.dem_encoder.modulate_features(pooled, scale, shift)

        # 5. Classify
        ci_logit = self.clf(pooled).squeeze(-1)  # (B,)
        ci_prob_pos = torch.sigmoid(ci_logit)  # (B,)
        ci_prob = torch.stack([1.0 - ci_prob_pos, ci_prob_pos], dim=-1)  # (B, 2)
        cognitive_impairment = (ci_prob_pos >= 0.5).long()  # (B,)

        ci_loss = None
        if "labels" in input_tensors:
            labels = input_tensors["labels"].to(self.device).to(self.dtype)
            ci_loss = self.criterion(ci_logit, labels)

        return {
            # (B, 1) — shaped for CINC2026Outputs compatibility
            "ci_logits": ci_logit.unsqueeze(-1),
            # (B, 2) — [:, 1] is P(CI=1); CINC2026Outputs.__post_init__ uses this
            "ci_prob": ci_prob,
            "cognitive_impairment": cognitive_impairment,
            "ci_loss": ci_loss,
        }

    # ─── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        epoch_features: Union[np.ndarray, torch.Tensor],
        demographics: Union[np.ndarray, torch.Tensor],
        padding_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> CINC2026Outputs:
        """Run inference on a single sample or a batch.

        Parameters
        ----------
        epoch_features : ndarray or Tensor, shape ``(T, caisr_feat_dim)`` or ``(B, T, caisr_feat_dim)``
            CAISR epoch feature vectors for one night or a batch of nights.
        demographics : ndarray or Tensor, shape ``(demographic_dim,)`` or ``(B, demographic_dim)``
            Normalised demographic features.
        padding_mask : ndarray or BoolTensor, shape ``(T,)`` or ``(B, T)``, optional
            Padding mask (True = padding position).

        Returns
        -------
        CINC2026Outputs
        """
        self.eval()

        def _to_tensor(v: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            return torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        epoch_features = _to_tensor(epoch_features)
        demographics = _to_tensor(demographics)

        # Handle single-sample input: add batch dimension
        if epoch_features.ndim == 2:
            epoch_features = epoch_features.unsqueeze(0)
            demographics = demographics.unsqueeze(0)
            if padding_mask is not None:
                padding_mask = _to_tensor(padding_mask).unsqueeze(0)

        tensors: Dict[str, torch.Tensor] = {
            "epoch_features": epoch_features,
            "demographics": demographics,
        }
        if padding_mask is not None:
            tensors["padding_mask"] = _to_tensor(padding_mask)

        output_dict = self.forward(tensors)
        return CINC2026Outputs.from_dict(output_dict)
