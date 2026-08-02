"""EpochCRNN: ResNet-N backbone + Bidirectional LSTM for CinC 2026.

Each PSG night is represented as a variable-length sequence of T × 30-second
epochs, each encoded as a CAISR feature vector.  The model treats this
``(B, T, caisr_feat_dim)`` input as a 1-D signal with **caisr_feat_dim**
channels and **T time-steps** (one per epoch), then applies a
:class:`torch_ecg.models.ECG_CRNN` backbone.

Architecture
------------
1. **Transpose**: ``(B, T, D)`` → ``(B, D, T)`` — CAISR features become the
   channel dimension.

2. **ResNet-N CNN** (epoch-scale variant, selected via ``config.cnn.name``):

   Example for the default ``resnetN_M`` backbone:

   .. code-block:: text

        Stem:    Conv1d(D→32, k=5, stride=1) + BN + ReLU
       Block 1: BasicBlock(32→ 32, k=5, stride=2)   T   → T/2
       Block 2: BasicBlock(32→ 64, k=3, stride=2)   T/2 → T/4
       Block 3: BasicBlock(64→128, k=3, stride=2)   T/4 → T/8
       CNN out: (B, 128, T/8) ≈ (B, 128, 96) for T=768

3. **Bidirectional LSTM** — ``retseq=False`` — summarises the temporal
   sequence into a single ``(B, 2·hidden)`` vector (last hidden state,
   both directions concatenated).

4. **FiLM demographic modulation** — :class:`~models.building_blocks.DemographicEncoder`
   with ``mode="film"`` conditions the feature on
   ``[age/100, sex (0/1), bmi/50]``.

5. **MLP classification head** — ``(2·hidden) → clf_hidden → 1`` scalar logit,
   trained with :class:`~torch.nn.BCEWithLogitsLoss`.

Size presets (set ``config = ModelCfg.epoch_crnn_<size>``)
----------------------------------------------------------
=========  =================  ============  =========  ~params
Preset     CNN channels       LSTM hidden   clf head   -------
=========  =================  ============  =========  ~params
``_S``     16 → 32 → 64       [64]          [32]       ~101 K
``_M``     32 → 64 → 128      [128]         [64]       ~437 K  *(default)*
``_L``     64 → 128 → 256     [256]         [128]      ~1.58 M
=========  =================  ============  =========  ~params

Swap backbone by changing ``config.cnn.name``; see
:data:`~model_configs.EPOCH_CRNN_CONFIG` for the full list of registered
backbone variants (``resnetN_S/M/L``, ``resnetNS_M``, ``resnetNB_M``).

Notes
-----
*No sampling-frequency concept applies here* — the "time" axis is the epoch
index (30 s/epoch), so all kernel sizes are dimensioned in epochs, not samples.
Kernel sizes of 3 and 5 correspond to 90-second and 150-second context windows.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.models import ECG_CRNN
from torch_ecg.models._nets import MLP
from torch_ecg.models.loss import setup_criterion

from cfg import ModelCfg
from outputs import CINC2026Outputs

from .building_blocks import DemographicEncoder

__all__ = ["EpochCRNN", "GradientReversalFunction", "AgeAdversarialHead"]


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for domain-adversarial training.

    ``forward`` is the identity; ``backward`` negates the incoming gradient
    (scaled by ``alpha``) so the upstream features are pushed *away* from
    predicting the adversarial target.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class AgeAdversarialHead(torch.nn.Module):
    """Age-regression head with gradient reversal.

    Attached to the pooled backbone representation; forces the backbone
    features to become age-invariant by minimising the *negated* age-loss
    gradient, so the main (CI) task can no longer exploit age as a shortcut.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.grl = GradientReversalFunction.apply
        self.head = MLP(
            in_channels=feature_dim,
            out_channels=[hidden_dim, 1],
            activation="gelu",
            bias=True,
            dropouts=0.1,
            skip_last_activation=True,
        )
        self.criterion = torch.nn.MSELoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.grl(features, self.alpha)
        return self.head(features)  # (B, 1) raw prediction of (age/100)


class EpochCRNN(ECG_CRNN):
    """ResNet-N + BiLSTM CRNN for the CinC 2026 Challenge.

    Parameters
    ----------
    config : CFG, optional
        Model configuration.  Defaults to ``ModelCfg.epoch_crnn``
        (alias for ``ModelCfg.epoch_crnn_M``).
        Pass ``ModelCfg.epoch_crnn_S`` or ``ModelCfg.epoch_crnn_L`` for
        the small / large size presets.  To swap the CNN backbone without
        changing the size class, set ``config.cnn.name`` to any key
        registered in :data:`~model_configs.EPOCH_CRNN_CONFIG`.
    **kwargs
        Any key overrides the corresponding field in ``config``.

    Input dict keys (``input_tensors``)
    ------------------------------------
    epoch_features : Tensor, shape ``(B, T, caisr_feat_dim)``
        CAISR epoch feature vectors (output of ``dataset.build_epoch_features``).
    demographics : Tensor, shape ``(B, demographic_dim)``
        Normalised demographic features ``[age/100, sex (0=F/1=M), bmi/50]``.
        Missing/invalid values default to ``[0.6, 0.0, 0.5]``.
    labels : Tensor, shape ``(B,)``, optional
        Binary ground-truth labels (float, 0 or 1) used for loss computation.

    Output dict keys
    ----------------
    ci_logits : Tensor, shape ``(B, 1)``
        Raw BCEWithLogitsLoss-compatible logit.  Shaped (B, 1) for
        :class:`~outputs.CINC2026Outputs` compatibility.
    ci_prob : Tensor, shape ``(B, 2)``
        ``[P(CI=0), P(CI=1)]`` derived via sigmoid.
    cognitive_impairment : LongTensor, shape ``(B,)``
        Hard prediction at threshold 0.5.
    ci_loss : Tensor or None
        Scalar BCEWithLogitsLoss value when ``labels`` are provided.

    Notes
    -----
    The ``padding_mask`` key accepted by :class:`~models.EpochTransformer` is
    **not** used here: the BiLSTM with ``retseq=False`` summarises the whole
    sequence into the last hidden state, so padding positions only add a small
    amount of spurious information.  For the typical 2–4 % padding fraction at
    ``max_seq_len=768`` this is negligible.  Future work could pack the sequence
    via :func:`torch.nn.utils.rnn.pack_padded_sequence` for exact masking.
    """

    __name__ = "EpochCRNN"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        if config is None:
            config = deepcopy(ModelCfg.epoch_crnn)
        else:
            config = deepcopy(config)
        config.update(kwargs)

        caisr_feat_dim: int = config.caisr_feat_dim  # 21

        # Pass the full nested config directly to ECG_CRNN.
        # ECG_CRNN reads config.cnn.name → config.cnn[name] for the backbone,
        # config.rnn.name → config.rnn[rnn_name] for the RNN, etc.
        # Extra keys (caisr_feat_dim, criterion, dem_encoder …) are ignored.
        super().__init__(
            classes=config.classes,
            n_leads=caisr_feat_dim,
            config=config,
        )
        # ECG_CRNN.__init__ overwrites self.config — restore our version.
        self.config = config
        self.classes = list(self.config.classes)
        self.n_classes = len(self.classes)

        # ── Replace 2-class softmax clf with 1-logit BCE clf ─────────────────
        # ECG_CRNN built a 2-output MLP; swap it for a single-logit head so we
        # train with BCEWithLogitsLoss (consistent with EpochTransformer).
        clf_in: int = self.clf.in_channels  # = 2 × rnn_hidden
        clf_hidden: list = list(self.config.clf.out_channels)  # e.g. [64]
        self.clf = MLP(
            in_channels=clf_in,
            out_channels=[*clf_hidden, 1],
            activation=self.config.clf.activation,
            bias=True,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        # ── FiLM demographic encoder ──────────────────────────────────────────
        dem_cfg = self.config.dem_encoder
        if dem_cfg.enable:
            self.dem_encoder = DemographicEncoder(
                dem_input_dim=dem_cfg.input_dim,
                feature_dim=clf_in,
                mode=dem_cfg.mode,
                hidden_dim=dem_cfg.hidden_dim,
            )
        else:
            self.dem_encoder = None

        # ── Criterion ─────────────────────────────────────────────────────────
        self.criterion = setup_criterion(self.config.criterion, **self.config.get("criterion_kw", {}))

        # ── Age-adversarial head (gradient reversal) ──────────────────────────
        # Config keys (all optional, defaults shown):
        #   age_adv.enable      (bool,  False) — toggle the branch
        #   age_adv.alpha       (float, 0.5)   — GRL gradient-reversal coefficient
        #   age_adv.lambda_     (float, 1.0)   — weight of the age MSE in the total loss
        #   age_adv.hidden_dim  (int,   32)    — age-head MLP width
        #   age_adv.position    ("before_film" | "after_film") — where the head taps
        #                          the representation.  "before_film" (default)
        #                          forces the *backbone* to be age-invariant while
        #                          FiLM remains the only explicit age channel.
        age_adv_cfg = self.config.get("age_adv", None)
        self.age_adv: Optional[AgeAdversarialHead] = None
        self.age_adv_lambda: float = 1.0
        self.age_adv_position: str = "before_film"
        if age_adv_cfg is not None and age_adv_cfg.get("enable", False):
            self.age_adv = AgeAdversarialHead(
                feature_dim=clf_in,
                hidden_dim=int(age_adv_cfg.get("hidden_dim", 32)),
                alpha=float(age_adv_cfg.get("alpha", 0.5)),
            )
            self.age_adv_lambda = float(age_adv_cfg.get("lambda_", 1.0))
            self.age_adv_position = str(age_adv_cfg.get("position", "before_film"))

    # ─── Forward pass ────────────────────────────────────────────────────────

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, None]]:
        """Forward pass accepting an ``input_tensors`` dict.

        Transposes ``epoch_features`` from ``(B, T, 21)`` to ``(B, 21, T)``
        before feeding it to the inherited :class:`ECG_CRNN` backbone.
        """
        # (B, T, caisr_dim) → (B, caisr_dim, T) for Conv1d backbone
        x = input_tensors["epoch_features"].to(self.device).to(self.dtype).transpose(1, 2)
        demographics = input_tensors["demographics"].to(self.device).to(self.dtype)

        # ── Backbone: CNN + BiLSTM (via ECG_CRNN) ────────────────────────────
        features = self.extract_features(x)  # (B, 2·rnn_hidden)
        features = self.pool(features)
        features = self.pool_rearrange(features)

        # ── Age-adversarial branch (before FiLM) ──────────────────────────────
        age_loss = None
        age_pred = None
        if self.age_adv is not None and self.age_adv_position == "before_film":
            age_pred = self.age_adv(features)  # (B, 1)
            age_target = demographics[:, 0:1]  # age/100
            age_loss = self.age_adv.criterion(age_pred, age_target)

        # ── FiLM demographic modulation ───────────────────────────────────────
        if self.dem_encoder is not None:
            scale, shift = self.dem_encoder(demographics)
            features = self.dem_encoder.modulate_features(features, scale, shift)

        # ── Age-adversarial branch (after FiLM) ───────────────────────────────
        if self.age_adv is not None and self.age_adv_position == "after_film":
            age_pred = self.age_adv(features)
            age_target = demographics[:, 0:1]
            age_loss = self.age_adv.criterion(age_pred, age_target)

        # ── Classify (1-logit BCE) ─────────────────────────────────────────────
        ci_logit = self.clf(features).squeeze(-1)  # (B,)
        ci_prob_pos = torch.sigmoid(ci_logit)  # (B,)
        ci_prob = torch.stack([1.0 - ci_prob_pos, ci_prob_pos], dim=-1)  # (B, 2)
        cognitive_impairment = (ci_prob_pos >= 0.5).long()  # (B,)

        ci_loss = None
        if "labels" in input_tensors:
            labels = input_tensors["labels"].to(self.device).to(self.dtype)
            ci_loss = self.criterion(ci_logit, labels)
            if age_loss is not None:
                ci_loss = ci_loss + self.age_adv_lambda * age_loss

        return {
            "ci_logits": ci_logit.unsqueeze(-1),  # (B, 1) for CINC2026Outputs
            "ci_prob": ci_prob,  # (B, 2)
            "cognitive_impairment": cognitive_impairment,
            "ci_loss": ci_loss,  # total loss (CI BCE + λ·age MSE) when labels present
            "age_pred": age_pred,  # (B, 1) or None
            "age_loss": age_loss,  # scalar or None
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
        padding_mask : ignored
            Accepted for API compatibility with :meth:`EpochTransformer.inference`
            but not used (see class docstring).

        Returns
        -------
        CINC2026Outputs
        """
        self.eval()

        def _to_tensor(v: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            return torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        epoch_features = _to_tensor(epoch_features)
        demographics = _to_tensor(demographics)

        if epoch_features.ndim == 2:
            epoch_features = epoch_features.unsqueeze(0)
            demographics = demographics.unsqueeze(0)

        tensors: Dict[str, torch.Tensor] = {
            "epoch_features": epoch_features,
            "demographics": demographics,
        }

        output_dict = self.forward(tensors)
        return CINC2026Outputs.from_dict(output_dict)
