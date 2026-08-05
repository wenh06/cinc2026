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
import torch.nn.functional as F
from torch_ecg.cfg import CFG
from torch_ecg.models import ECG_CRNN
from torch_ecg.models._nets import MLP
from torch_ecg.models.loss import setup_criterion

from cfg import ModelCfg
from const import NIGHT_FEATURE_DIM
from outputs import CINC2026Outputs

from .building_blocks import DemographicEncoder

__all__ = [
    "EpochCRNN",
    "GradientReversalFunction",
    "AgeAdversarialHead",
    "FocalBCEWithLogitsLoss",
    "AgeMatchedPairwiseLossHinge",
]


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


class FocalBCEWithLogitsLoss(torch.nn.Module):
    """Focal loss on top of :class:`~torch.nn.BCEWithLogitsLoss`.

    ``FL = (1 - p_t)^gamma * BCE`` with ``p_t = exp(-BCE)``, where BCE uses
    the same ``pos_weight`` as the baseline criterion.  Keeps the
    ``pos_weight`` semantics of the O0 baseline (class imbalance handled by
    the existing bridge in ``team_code._train_single_fold``); the ``gamma``
    down-weights easy samples so the loss focuses on hard positives/negatives.

    Parameters
    ----------
    gamma : float, default 2.0
        Focusing parameter; larger values down-weight easy samples more.
    pos_weight : torch.Tensor or float, optional
        Weight for the positive class (BCEWithLogitsLoss semantics).
    reduction : {"mean", "sum", "none"}, default "mean"

    """

    __name__ = "FocalBCEWithLogitsLoss"

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[Union[torch.Tensor, float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )  # (B,)
        p_t = torch.exp(-ce)  # "probability" of being correct per BCE
        fl = (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl


class AgeMatchedPairwiseLossHinge(torch.nn.Module):
    """Age-matched pairwise margin loss — direct optimisation proxy of the
    official primary metric (age-conditioned AUROC).

    Forms positive-negative pairs whose ages differ by ≤ ``tolerance`` years
    (the same pairing rule as the official ``age_conditioned_auroc``) across
    the current batch and a rolling FIFO memory bank of recent samples, and
    minimises ``mean(max(0, margin - (s_pos - s_neg)))``.  Bank logits are
    stored detached, so gradients flow only through the current batch.

    Parameters
    ----------
    margin : float, default 0.5
        Hinge margin.
    tolerance : float, default 2.0
        Max |age difference| in *years* for a valid pair (ages are fed in
        ``age/100`` units, matching ``demographics[:, 0]``).
    bank_size : int, default 512
        Memory-bank capacity (rolling FIFO).  With the official-phase 7.6 %
        prevalence and batch_size 16, batch-internal pairing yields ~1
        positive per batch — the bank supplies the negative (and positive)
        mass that makes age-matched pairing meaningful.

    """

    __name__ = "AgeMatchedPairwiseLossHinge"

    def __init__(self, margin: float = 0.5, tolerance: float = 2.0, bank_size: int = 512) -> None:
        super().__init__()
        self.margin = float(margin)
        self.tolerance = float(tolerance) / 100.0  # years → age/100 units
        self.bank_size = int(bank_size)
        self._bank: list = []  # rolling FIFO of (logit, age/100, label) floats

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        ages: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        scores : torch.Tensor
            Raw logits (B,), gradient-carrying for the current batch.
        labels : torch.Tensor
            Binary labels (B,), float 0/1.
        ages : torch.Tensor
            Ages in ``age/100`` units (B,), matching ``demographics[:, 0]``.

        Returns
        -------
        torch.Tensor
            Mean hinge loss over all valid age-matched pairs; zero (but
            differentiable) when no valid pair exists.

        """
        device, dtype = scores.device, scores.dtype

        # Pairing pool = current batch (gradient-carrying) + bank (constants).
        # The bank update happens AFTER pairing — pushing the current batch
        # into the bank first would duplicate it in the pool (once with
        # gradients, once detached) and dilute the loss mean with no-gradient
        # duplicate pairs.
        bank_scores = torch.tensor([t[0] for t in self._bank], device=device, dtype=dtype)
        bank_ages = torch.tensor([t[1] for t in self._bank], device=device, dtype=dtype)
        bank_labels = torch.tensor([t[2] for t in self._bank], device=device, dtype=dtype)
        all_scores = torch.cat([scores, bank_scores])
        all_ages = torch.cat([ages, bank_ages])
        all_labels = torch.cat([labels, bank_labels])

        pos_idx = torch.nonzero(all_labels == 1.0).squeeze(-1)  # (P,)
        neg_idx = torch.nonzero(all_labels == 0.0).squeeze(-1)  # (N,)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            self._update_bank(scores, ages, labels)
            return scores.sum() * 0.0  # keep differentiable

        age_diff = (all_ages[pos_idx][:, None] - all_ages[neg_idx][None, :]).abs()  # (P, N)
        valid = age_diff <= self.tolerance
        if not valid.any():
            self._update_bank(scores, ages, labels)
            return scores.sum() * 0.0

        pos_sel, neg_sel = torch.nonzero(valid, as_tuple=True)
        diff = self.margin - (all_scores[pos_idx[pos_sel]] - all_scores[neg_idx[neg_sel]])
        loss = F.relu(diff).mean()

        # Rolling FIFO bank update — store detached floats so the bank never
        # carries a stale autograd graph.  Called on every path (incl. the
        # early returns) so the bank always accumulates across steps.
        self._update_bank(scores, ages, labels)

        return loss

    def _update_bank(
        self,
        scores: torch.Tensor,
        ages: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Push the current batch into the rolling FIFO bank (detached floats)."""
        self._bank.extend(
            zip(
                scores.detach().tolist(),
                ages.detach().tolist(),
                labels.detach().tolist(),
            )
        )
        if len(self._bank) > self.bank_size:
            del self._bank[: len(self._bank) - self.bank_size]


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

        # ── Night-level aggregation features (P1, Phase 9) ────────────────────
        # Late fusion: the backbone output (B, 2·rnn_hidden) is concatenated
        # with the night-MLP output (B, hidden_dim[-1]) BEFORE FiLM.  The
        # age-adv "before_film" tap keeps seeing the raw backbone output, so
        # its head dim is unchanged.  When disabled, night_out_dim = 0 →
        # fused_dim == clf_in → the architecture is bit-identical to the O0
        # baseline.  Config keys (all optional, defaults shown):
        #   night_features.enable     (bool,  False)   — toggle the branch
        #   night_features.dim        (int,   15)      — fixed by dataset.build_night_features
        #   night_features.hidden_dim ([int], [32,16]) — MLP widths; [-1] = fusion dim
        #   night_features.activation (str,   "gelu")
        #   night_features.dropouts   (float, 0.1)
        night_cfg = self.config.get("night_features", None)
        self.night_encoder: Optional[MLP] = None
        night_out_dim: int = 0
        if night_cfg is not None and night_cfg.get("enable", False):
            night_dim = int(night_cfg.get("dim", 15))
            assert (
                night_dim == NIGHT_FEATURE_DIM
            ), f"night feature dim fixed at {NIGHT_FEATURE_DIM} by build_night_features, got {night_dim}"
            self.night_encoder = MLP(
                in_channels=night_dim,
                out_channels=list(night_cfg.get("hidden_dim", [32, 16])),
                activation=str(night_cfg.get("activation", "gelu")),
                bias=True,
                dropouts=float(night_cfg.get("dropouts", 0.1)),
            )
            night_out_dim = list(night_cfg.get("hidden_dim", [32, 16]))[-1]  # e.g. 16

        # ── Replace 2-class softmax clf with 1-logit BCE clf ─────────────────
        # ECG_CRNN built a 2-output MLP; swap it for a single-logit head so we
        # train with BCEWithLogitsLoss (consistent with EpochTransformer).
        clf_in: int = self.clf.in_channels  # = 2 × rnn_hidden
        fused_dim: int = clf_in + night_out_dim  # + night fusion dim when enabled
        clf_hidden: list = list(self.config.clf.out_channels)  # e.g. [64]
        self.clf = MLP(
            in_channels=fused_dim,
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
                feature_dim=fused_dim,
                mode=dem_cfg.mode,
                hidden_dim=dem_cfg.hidden_dim,
            )
        else:
            self.dem_encoder = None

        # ── Criterion ─────────────────────────────────────────────────────────
        # FocalBCEWithLogitsLoss is a local custom loss (keeps the baseline
        # pos_weight semantics); everything else goes through torch_ecg's
        # setup_criterion (BCEWithLogitsLoss, FocalLoss, ...).
        if self.config.criterion == "FocalBCEWithLogitsLoss":
            self.criterion = FocalBCEWithLogitsLoss(**self.config.get("criterion_kw", {}))
        else:
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

        # ── O4: drop the age channel from FiLM demographics ─────────────────────
        # Config key: no_age (bool, False).  When enabled, demographics
        # [age/100, sex, bmi/50] become [0, sex, bmi/50] in forward — a constant
        # channel carries no information, so the FiLM modulator learns a fixed
        # per-(sex,bmi) modulation and the model cannot exploit age as a
        # between-stratum shortcut (the official metric is age-conditioned
        # AUROC).  Mutually exclusive with age_adv, whose regression target is
        # the age channel.
        self.no_age = bool(self.config.get("no_age", False))
        if self.no_age and self.age_adv is not None:
            raise ValueError("no_age and age_adv are mutually exclusive: age_adv needs the age channel")

        # ── O7: age-matched pairwise ranking loss ──────────────────────────────
        # Direct optimisation proxy of the official primary metric (age-
        # conditioned AUROC): adds λ·mean(max(0, margin − (s_pos − s_neg)))
        # over positive-negative pairs with |age diff| ≤ tolerance, assembled
        # across the current batch + a rolling memory bank (see
        # AgeMatchedPairwiseLossHinge).  Config keys (all optional):
        #   enable     (bool,  False) — toggle the branch
        #   lambda_    (float, 1.0)   — weight of the pairwise loss
        #   margin     (float, 0.5)   — hinge margin
        #   tolerance  (float, 2.0)   — max |age diff| (years), matches the
        #                              official age_conditioned_auroc
        #   bank_size  (int,   512)   — memory-bank capacity
        ap_cfg = self.config.get("age_pairwise", None)
        self.age_pairwise: Optional[AgeMatchedPairwiseLossHinge] = None
        self.ap_lambda: float = 0.0
        if ap_cfg is not None and ap_cfg.get("enable", False):
            if self.no_age:
                raise ValueError("age_pairwise and no_age are mutually exclusive: pairwise needs the age channel")
            self.age_pairwise = AgeMatchedPairwiseLossHinge(
                margin=float(ap_cfg.get("margin", 0.5)),
                tolerance=float(ap_cfg.get("tolerance", 2.0)),
                bank_size=int(ap_cfg.get("bank_size", 512)),
            )
            self.ap_lambda = float(ap_cfg.get("lambda_", 1.0))

    # ─── Forward pass ────────────────────────────────────────────────────────

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, None]]:
        """Forward pass accepting an ``input_tensors`` dict.

        Transposes ``epoch_features`` from ``(B, T, 21)`` to ``(B, 21, T)``
        before feeding it to the inherited :class:`ECG_CRNN` backbone.
        """
        # (B, T, caisr_dim) → (B, caisr_dim, T) for Conv1d backbone
        x = input_tensors["epoch_features"].to(self.device).to(self.dtype).transpose(1, 2)
        demographics = input_tensors["demographics"].to(self.device).to(self.dtype)
        if self.no_age:
            # O4: constant age channel → informationally equivalent to removing
            # it; the FiLM modulator sees only (0, sex, bmi/50).
            demographics = torch.cat([torch.zeros_like(demographics[:, :1]), demographics[:, 1:]], dim=-1)

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

        # ── Night-level aggregation features (late fusion, before FiLM) ───────
        # The night-MLP output is concatenated to the backbone output here, so
        # FiLM modulates the fused vector and the "after_film" age-adv tap (if
        # selected) sees the night contribution as well.
        if self.night_encoder is not None:
            night = self.night_encoder(input_tensors["night_features"].to(self.device).to(self.dtype))  # (B, 16)
            features = torch.cat([features, night], dim=-1)  # (B, 2·rnn_hidden + 16)

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
        cognitive_impairment = (ci_prob_pos >= self.config.get("binary_threshold", 0.5)).long()  # (B,)

        ci_loss = None
        if "labels" in input_tensors:
            labels = input_tensors["labels"].to(self.device).to(self.dtype)
            ci_loss = self.criterion(ci_logit, labels)
            # O7: age-matched pairwise ranking loss (official primary-metric
            # proxy).  Training only — the memory bank must not be polluted
            # by evaluation passes (model.eval() → self.training == False).
            # Uses the RAW (unsmoothed) labels: run_one_step smooths
            # ``labels`` when label_smoothing > 0, and the pairing masks
            # (== 1.0 / == 0.0) would silently match nothing on smoothed
            # targets.
            if self.age_pairwise is not None and self.training:
                ap_labels = input_tensors.get("label", labels).to(self.device).to(self.dtype)
                ap_loss = self.age_pairwise(
                    ci_logit,
                    ap_labels,
                    input_tensors["demographics"].to(self.device).to(self.dtype)[:, 0],
                )
                ci_loss = ci_loss + self.ap_lambda * ap_loss
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
        night_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
        night_features : ndarray or Tensor, shape ``(night_feature_dim,)`` or ``(B, night_feature_dim)``, optional
            Night-level aggregation features (P1).  Only used when the model's
            night branch is enabled; ignored otherwise (so old checkpoints and
            disabled models never touch this key).

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
        if self.night_encoder is not None and night_features is not None:
            night = _to_tensor(night_features)
            if night.ndim == 1:
                night = night.unsqueeze(0)
            tensors["night_features"] = night

        output_dict = self.forward(tensors)
        return CINC2026Outputs.from_dict(output_dict)
