"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG

from const import (
    BINARY_AROUSAL_FEATURE_SET,
    NIGHT_FEATURE_DIM,
    resolve_feature_pipeline,
)
from model_configs import EPOCH_CRNN_CONFIG, EPOCH_TRANSFORMER_BASE

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
    "sync_feature_config",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent

###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.checkpoints = _BASE_DIR / "checkpoints"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.checkpoints.mkdir(exist_ok=True)

BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32

# CinC 2026 Specifics
BaseCfg.fs = 100  # Standardized sampling frequency for processing
BaseCfg.classes = ["Negative", "Positive"]
BaseCfg.num_classes = len(BaseCfg.classes)
BaseCfg.demographic_features = ["Age", "Sex", "BMI"]


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

# Data Loader Configs
# batch_size=16 balances gradient noise and OneCycleLR step count.
TrainCfg.batch_size = 16
# train_ratio: fallback 80/20 split used by CINC2026Dataset when the canonical
# JSON split file (utils/cinc2026-data-split.json) is absent.
TrainCfg.train_ratio = 0.8

# folds: 5-fold CV ensemble.  None = single model on the canonical split;
# [0..4] = one model per fold (utils/cinc2026-5fold-split.json) saved to
# model_folder/fold_{k}/; probabilities averaged at inference (binary =
# majority vote).  Official re-training runs len(folds) × single-fold time.
# Sub3: default = 5-fold — fits the official 72-h budget, averages out the
# ~0.08 per-fold age-cond spread.
TrainCfg.folds = [0, 1, 2, 3, 4]
TrainCfg.model_name = "epoch_crnn_M"  # best AUROC so far (sub3: 0.555)
TrainCfg.feature_set = BINARY_AROUSAL_FEATURE_SET  # submission-1~4 feature set; best public run so far

# learning_rate is the canonical name used by BaseTrainer; lr is kept as an alias
TrainCfg.lr = 3e-4
TrainCfg.learning_rate = TrainCfg.lr

# db_dir must be set at training time (e.g. via command-line argument)
TrainCfg.db_dir = None

# Monitor metric for model selection and early stopping.
# Official phase primary metric: age-conditioned AUROC ("auroc_age_cond"),
# computed over positive-negative pairs within ±2 years of age; falls back to
# plain AUROC when no valid age-matched pair exists on the evaluation split.
TrainCfg.monitor = "auroc_age_cond"

# Misc training flags
TrainCfg.debug = False
TrainCfg.flooding_level = 0  # no flooding regularisation by default

# Epoch-sequence settings.
# max_seq_len: cap the number of 30s epochs per record.  768 epochs ≈ 6.4 h,
# covering the majority of recording lengths while keeping training step time
# uniform.  During training a random crop is applied; during validation the
# centre crop is used (see FastDataReader).  Set to None to always process the
# full sequence.
TrainCfg.max_seq_len = 768

# sig_len: raw-signal compatibility knob kept for models/transformer.py
# prototype.  Not used in the CAISR epoch-feature pipeline.
TrainCfg.sig_len = 3000  # 30 seconds at 100Hz

# Optimization Configs.
TrainCfg.n_epochs = 100
TrainCfg.optimizer = "adamw_amsgrad"
# weight_decay: BaseTrainer._setup_optimizer reads get_kwargs(AdamW) which uses the
# key "weight_decay" — NOT "decay".  Always use "weight_decay" here.
TrainCfg.weight_decay = 1e-2
TrainCfg.lr_scheduler = "one_cycle"
TrainCfg.max_lr = 1e-3  # OneCycleLR peak
# pct_start: OneCycleLR warm-up fraction (0.3 Transformer / 0.1 CRNN).
TrainCfg.pct_start = 0.3
TrainCfg.betas = (0.9, 0.999)
TrainCfg.grad_clip = 1.0  # gradient clipping max norm (0 to disable)

# Augmentation / Regularisation
# CAISR features are pre-bounded in [0,1]; PreprocManager is for raw signals only.
#
# label_smoothing: targets {0,1} → {ε/2, 1-ε/2}.  0.05 with focal (O7b) —
# measured +0.049 age-cond on the full-train eval (0.8018 vs 0.7525).
TrainCfg.label_smoothing = 0.05

# pos_weight: BCEWithLogitsLoss pos_weight for the minority (CI-positive) class.
# Official phase prevalence is 7.6% → positive:negative ≈ 1:12, so we up-weight
# positives by ~12× to keep the loss from being dominated by the negative class.
TrainCfg.pos_weight = 12.16  # ≈ (1-0.076)/0.076

# Per-record z-score normalization; off by default (submission-5 regression).
# Enable via TrainCfg.normalize = CFG(method="per_record_zscore", ...).
TrainCfg.normalize = None

# Age-adversarial head (gradient reversal) — P0 age-dependence mitigation.
# When enabled, the pooled backbone representation is fed through a gradient-
# reversal layer into an age-regression head; the reversed gradient forces the
# backbone to become age-invariant, leaving FiLM as the only explicit age
# channel.  Keys mirror the model config (see models/epoch_crnn.py):
#   enable      (bool,  False) — toggle the branch
#   alpha       (float, 0.5)   — GRL coefficient
#   lambda_     (float, 1.0)   — weight of the age MSE in the total loss
#   hidden_dim  (int,   32)    — age-head MLP width
#   position    ("before_film" | "after_film")
TrainCfg.age_adv = CFG(
    enable=False,
    alpha=0.5,
    lambda_=1.0,
    hidden_dim=32,
    position="before_film",
)

# Night-level aggregation features (P1, Phase 9) — per-night 15-dim summary
# statistics computed from the FULL-night CAISR annotations (independent of
# the per-epoch matrix, which is cropped to max_seq_len=768).  Fused into the
# CRNN backbone output via late concatenation: backbone (B, 2·rnn_hidden)
# → concat MLP(15→32→16) output (B, 16) → FiLM → clf.  Keys mirror the model
# config (see models/epoch_crnn.py):
#   enable      (bool,  False)      — toggle the branch (default OFF = O0 baseline)
#   dim         (int,   15)         — fixed by dataset.build_night_features
#   hidden_dim  ([int], [32, 16])   — MLP widths; hidden_dim[-1] = fusion dim
#   activation  (str,   "gelu")
#   dropouts    (float, 0.1)
TrainCfg.night_features = CFG(
    enable=False,
    dim=NIGHT_FEATURE_DIM,
    hidden_dim=[32, 16],
    activation="gelu",
    dropouts=0.1,
)

# O4: zero the age channel in FiLM demographics (age is constant within each
# stratum → can only power between-stratum shortcuts).  EpochCRNN only;
# mutually exclusive with age_adv (needs the age channel).
TrainCfg.no_age = False

# O7: age-matched pairwise ranking loss — direct age-cond AUROC proxy:
# λ·hinge(margin − (s_pos − s_neg)) over pos/neg pairs with |age diff| ≤
# tolerance, via a rolling memory bank (see AgeMatchedPairwiseLossHinge).
#   enable     (bool,  False) — toggle (default OFF = O0 baseline)
#   lambda_    (float, 1.0)   — pairwise weight
#   margin     (float, 0.5)   — hinge margin
#   tolerance  (float, 2.0)   — max |age diff| years (matches the official metric)
#   bank_size  (int,   512)   — memory-bank capacity
TrainCfg.age_pairwise = CFG(
    enable=False,
    lambda_=1.0,
    margin=0.5,
    tolerance=2.0,
    bank_size=512,
)

# O7: focal loss — (1−pt)^γ·BCE (pt = exp(−BCE), same pos_weight as baseline)
# down-weights easy samples; enable → team_code swaps the criterion to
# FocalBCEWithLogitsLoss.  Sub3: ON — focal + label_smoothing 0.05 measured
# +0.049 age-cond (full-train 0.8018 vs 0.7525).
#   enable  (bool,  True)  — toggle
#   gamma   (float, 2.0)
TrainCfg.focal = CFG(
    enable=True,
    gamma=2.0,
)

# D2 tabular pipeline — 641-dim spectral/physiological feature bank + boosted
# trees, an alternate submission path that bypasses the CRNN entirely.
# Default ON for sub5 (SMALL training set): XGBoost on the 390-dim spec block,
# metadata excluded.  include_meta stays False because meta_rec_year is a
# follow-up-window eligibility artifact (records from ~2019+ are almost all
# positive) that inflates the I0006/S0001/I0002 holdouts but does not transfer
# to the official val/test sites (I0004 2004-2016, I0007 2011-2017).
#   enable         (bool, True) — route train_model/run_model through the tabular branch
#   model          (str, "xgboost") — {"xgboost", "lightgbm"}
#   feature_groups (list, ["spec"]) — which blocks of the 641-dim bank to use
#                   {"spec","coh","tp","trans","arch","hrv","spo2"}; empty = all
#   include_meta   (bool, False) — append age / sex / bmi / recording-year
#   lgbm_params    (CFG) — LightGBM hyperparameters
#   xgb_params     (CFG) — XGBoost hyperparameters
#   feature_cache  (str, "") — precomputed features.csv (D1 layout, index =
#                   BidsFolder or BidsFolder__SessionID); cache-first, on-the-fly
#                   extraction from raw+CAISR for misses
#   workers        (int, 4) — on-the-fly extraction parallelism
#   threshold      (float, 0.5) — binary cutoff (Reward side only; age-cond is rank-based)
# D2-small reading (15 seeds, I0006-holdout): XGB spec-only 0.6446±0.0144,
# LGBM spec-only 0.6555±0.0230 vs small-pool CRNN 0.546; S0001-holdout XGB
# 0.6473±0.0076 — see ROADMAP experiment log rows D2-tabular / D2-large.
TrainCfg.tabular = CFG(
    enable=True,
    model="xgboost",
    feature_groups=["spec"],
    include_meta=False,
    lgbm_params=CFG(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        seed=0,
    ),
    xgb_params=CFG(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        seed=0,
    ),
    feature_cache="",
    workers=4,
    threshold=0.5,
)

# Phi component — frozen Philosopher's Stone latents -> PCA -> tabular ranker
# (the sub6 primary).  Latents are cache-first from `cache` (vendored at
# `data/phi_cache` in the image; resolves automatically when unset); misses are
# computed on the fly from the raw EDF using the baked checkpoint.  Records
# without a usable C4-M1 raise at inference and fall through to the next
# component.  I0006 proxy: lr_pca 0.6825, xgb_pca 0.6807 ± 0.0247 (PCA-64,
# latents only) — see scripts/phi_pca_ranker.py and tmp/phi_cache/results_pca.json.
#   model      (str, "ensemble") — {"xgboost", "logistic", "ensemble"};
#              "ensemble" averages the LR and XGB probability estimates
#   pca_dim    (int, 64)
#   include_scores (bool, False) — append the 4 brain-health scores to the PCA features
#   cache      (str, "") — latent cache dir; resolved to data/phi_cache when unset
#   checkpoint (str, "") — resolved via MODEL_CACHE_DIR when unset
#   xgb_params (CFG) / lr_params (CFG) — ranker hyperparameters
#   threshold  (float, 0.5) — binary cutoff (Reward side only; age-cond is rank-based)
TrainCfg.phi = CFG(
    enable=True,
    model="ensemble",
    pca_dim=64,
    include_scores=False,
    cache="",
    checkpoint="",
    xgb_params=CFG(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        seed=0,
    ),
    lr_params=CFG(
        C=1.0,
        max_iter=2000,
    ),
    threshold=0.5,
)

# Model-component registry — a submission may bundle several independently
# trained models with a per-record fallback chain.  Components run in
# `priority` order (lower first); a component that raises on a record falls
# through to the next one.  Each component trains into
# `model_folder/components/<name>/` and is recorded in
# `model_folder/model_manifest.json`, so `load_model` routes from the
# on-disk manifest rather than the current `TrainCfg`.
#   name     (str) — unique component name (also the artifact subfolder)
#   type     (str) — {"tabular", "crnn", "phi"}
#   enable   (bool, True) — train/load this component
#   priority (int, 0) — fallback order; lower runs first
# The sub6 default is phi_pca64 (priority 0) with sub5_tabular_xgb as the
# montage-agnostic fallback (priority 1).  `TrainCfg.tabular.enable` /
# `TrainCfg.phi.enable` are kept for the legacy no-components layout (see
# team_code.train_model / load_model).
TrainCfg.components = [
    CFG(
        name="phi_pca64",
        type="phi",
        enable=True,
        priority=0,
    ),
    CFG(
        name="sub5_tabular_xgb",
        type="tabular",
        enable=True,
        priority=1,
    ),
]

# Callbacks & Logging
TrainCfg.log_step = 20
TrainCfg.keep_checkpoint_max = 5
TrainCfg.early_stopping = CFG(
    min_delta=0.001,
    patience=15,  # with 100 epochs; stops ~15 epochs after last improvement
    min_epochs=30,  # early-stop countdown starts at epoch 30 (30% of 100);
    # prevents early lucky-spike bests (e.g. fold_4 best@ep9) from
    # triggering a premature stop while the curve is still climbing
)


###############################################################################
# configurations for building deep learning models
###############################################################################

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = BaseCfg.fs
_BASE_MODEL_CONFIG.classes = BaseCfg.classes
_BASE_MODEL_CONFIG.num_classes = BaseCfg.num_classes

ModelCfg = deepcopy(_BASE_MODEL_CONFIG)


# ── Helper: build one EpochTransformer size preset ───────────────────────────
def _make_epoch_transformer(d_model: int, nhead: int, num_layers: int, dim_feedforward: int) -> CFG:
    cfg = deepcopy(_BASE_MODEL_CONFIG)
    cfg.update(deepcopy(EPOCH_TRANSFORMER_BASE))
    cfg.d_model = d_model
    cfg.nhead = nhead
    cfg.num_layers = num_layers
    cfg.dim_feedforward = dim_feedforward
    # binary_threshold: P(CI=1) cutoff for binary predictions; serialised into
    # the checkpoint and overwritten with the val-tuned optimum after training.
    cfg.binary_threshold = 0.5
    return cfg


# EpochTransformer — operates on a sequence of 30-s-epoch CAISR feature vectors.
# Input:  (B, T, caisr_feat_dim) + demographics (B, demographic_dim)
# Output: binary classification (cognitive impairment)
#
# Three size presets (swap by setting TrainCfg.model_name):
#   epoch_transformer_S  →  64-dim,  2 heads, 2 layers, ff=256   ~110 K params
#   epoch_transformer_M  → 128-dim,  4 heads, 4 layers, ff=512   ~825 K params
#   epoch_transformer_L  → 256-dim,  8 heads, 6 layers, ff=1024  ~5.3 M params
ModelCfg.epoch_transformer_S = _make_epoch_transformer(
    d_model=64,
    nhead=2,
    num_layers=2,
    dim_feedforward=256,
)
ModelCfg.epoch_transformer_M = _make_epoch_transformer(
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
)
ModelCfg.epoch_transformer_L = _make_epoch_transformer(
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
)
# Canonical alias (M is the default)
ModelCfg.epoch_transformer = ModelCfg.epoch_transformer_M


# ── Helper: build one EpochCRNN size preset ───────────────────────────────────
def _make_epoch_crnn(cnn_name: str, lstm_hidden: list, clf_hidden: list) -> CFG:
    cfg = deepcopy(_BASE_MODEL_CONFIG)
    cfg.update(deepcopy(EPOCH_CRNN_CONFIG))
    cfg.caisr_feat_dim = 21
    cfg.demographic_dim = 3
    cfg.criterion = "BCEWithLogitsLoss"
    cfg.dem_encoder = CFG(enable=True, input_dim=3, hidden_dim=64, mode="film")
    cfg.age_adv = deepcopy(TrainCfg.age_adv)  # disabled by default; toggle at train time
    cfg.night_features = deepcopy(TrainCfg.night_features)  # disabled by default; toggle at train time
    cfg.no_age = deepcopy(TrainCfg.no_age)  # disabled by default; toggle at train time
    cfg.age_pairwise = deepcopy(TrainCfg.age_pairwise)  # disabled by default; toggle at train time
    # binary_threshold: P(CI=1) cutoff for binary predictions; serialised into
    # the checkpoint and overwritten with the val-tuned optimum after training.
    cfg.binary_threshold = 0.5
    # Select backbone
    cfg.cnn.name = cnn_name
    # Override LSTM hidden sizes for this preset
    cfg.rnn.lstm.hidden_sizes = list(lstm_hidden)
    # Override clf intermediate layers for this preset
    cfg.clf.out_channels = list(clf_hidden)
    return cfg


# EpochCRNN — ResNet-N backbone + BiLSTM for the epoch-feature sequence.
# Input:  (B, T, caisr_feat_dim) treated as (B, 21, T) for Conv1d
# Output: binary classification (cognitive impairment)
#
# Three size presets (swap by setting TrainCfg.model_name):
#   epoch_crnn_S  →  CNN [16→32→64],   LSTM hidden=[64],   clf=[32]   ~101 K params
#   epoch_crnn_M  →  CNN [32→64→128],  LSTM hidden=[128],  clf=[64]   ~437 K params
#   epoch_crnn_L  →  CNN [64→128→256], LSTM hidden=[256],  clf=[128]  ~1.58 M params
#
# Additional backbone swaps (same channel widths as M, different block types):
#   epoch_crnn_S/M/L with cnn.name = "resnetNS_M"  — separable convolutions
#   epoch_crnn_S/M/L with cnn.name = "resnetNB_M"  — bottleneck residual blocks
ModelCfg.epoch_crnn_S = _make_epoch_crnn("resnetN_S", lstm_hidden=[64], clf_hidden=[32])
ModelCfg.epoch_crnn_M = _make_epoch_crnn("resnetN_M", lstm_hidden=[128], clf_hidden=[64])
ModelCfg.epoch_crnn_L = _make_epoch_crnn("resnetN_L", lstm_hidden=[256], clf_hidden=[128])
# Canonical alias (M is the default)
ModelCfg.epoch_crnn = ModelCfg.epoch_crnn_M

# resnetNC_BNse presets — 4-stage bottleneck+SE backbone (Nature-Comm style)
# CNN out channels = num_filters[-1] * 4 (bottleneck expansion):
#   _BNse_S → 256 ch   _BNse_M → 512 ch   _BNse_L → 1024 ch
# Total params (CNN + BiLSTM + clf):  ~0.5M / ~1.9M / ~7.3M
ModelCfg.epoch_crnn_resnetNC_BNse_S = _make_epoch_crnn("resnetNC_BNse_S", lstm_hidden=[64], clf_hidden=[32])
ModelCfg.epoch_crnn_resnetNC_BNse_M = _make_epoch_crnn("resnetNC_BNse_M", lstm_hidden=[128], clf_hidden=[64])
ModelCfg.epoch_crnn_resnetNC_BNse_L = _make_epoch_crnn("resnetNC_BNse_L", lstm_hidden=[256], clf_hidden=[128])

# tresnetE presets — 4-stage TResNet-style backbone (mixed basic+bottleneck+SE)
# CNN out channels = num_filters[-1] * 4 (last two stages are bottleneck):
#   _S → 512 ch   _M → 1024 ch   _L → 1536 ch
# Total params (CNN + BiLSTM + clf):  ~0.7M / ~2.5M / ~7.4M
ModelCfg.epoch_crnn_tresnetE_S = _make_epoch_crnn("tresnetE_S", lstm_hidden=[64], clf_hidden=[32])
ModelCfg.epoch_crnn_tresnetE_M = _make_epoch_crnn("tresnetE_M", lstm_hidden=[128], clf_hidden=[64])
ModelCfg.epoch_crnn_tresnetE_L = _make_epoch_crnn("tresnetE_L", lstm_hidden=[256], clf_hidden=[128])


# ── Raw-signal prototype configs (NOT currently used) ────────────────────────
# These use CrossEntropyLoss (softmax over 2 classes) which is architecturally
# inconsistent with the BCEWithLogitsLoss used by EpochCRNN and EpochTransformer.
# They exist as scaffolding for a future raw-signal branch.
ModelCfg.transformer = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.transformer.d_model = 128
ModelCfg.transformer.nhead = 4
ModelCfg.transformer.num_layers = 4
ModelCfg.transformer.dim_feedforward = 512
ModelCfg.transformer.dropout = 0.1
ModelCfg.transformer.activation = "relu"
ModelCfg.transformer.max_channels = 25  # Max unique channels expected
ModelCfg.transformer.criterion = "CrossEntropyLoss"
ModelCfg.transformer.dem_encoder = CFG(enable=True, input_dim=len(BaseCfg.demographic_features), hidden_dim=64, mode="film")

# MultiBranchNet configuration (raw-signal fallback)
ModelCfg.multibranch = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.multibranch.d_model = 128
ModelCfg.multibranch.modalities = ["eeg", "eog", "emg", "ecg", "resp"]
ModelCfg.multibranch.nhead = 4
ModelCfg.multibranch.dropout = 0.1
ModelCfg.multibranch.criterion = "CrossEntropyLoss"
ModelCfg.multibranch.dem_encoder = deepcopy(ModelCfg.transformer.dem_encoder)


def sync_feature_config(train_cfg: CFG = TrainCfg, model_cfg: CFG = ModelCfg) -> None:
    """Synchronize feature-set-dependent dimensions and flags across configs."""
    pipeline = resolve_feature_pipeline(train_cfg.feature_set, train_cfg.model_name)
    train_cfg.include_time_encoding = pipeline["include_time_encoding"]
    train_cfg.caisr_feat_dim = pipeline["feature_dim"]

    if train_cfg.normalize and getattr(train_cfg.normalize, "method", "") == "per_record_zscore":
        train_cfg.normalize.skip_cols = pipeline["time_cols"]

    transformer_dim = resolve_feature_pipeline(train_cfg.feature_set, "epoch_transformer")["feature_dim"]
    crnn_dim = resolve_feature_pipeline(train_cfg.feature_set, "epoch_crnn")["feature_dim"]

    for name in [
        "epoch_transformer_S",
        "epoch_transformer_M",
        "epoch_transformer_L",
        "epoch_transformer",
    ]:
        getattr(model_cfg, name).caisr_feat_dim = transformer_dim

    for name in [
        "epoch_crnn_S",
        "epoch_crnn_M",
        "epoch_crnn_L",
        "epoch_crnn",
        "epoch_crnn_resnetNC_BNse_S",
        "epoch_crnn_resnetNC_BNse_M",
        "epoch_crnn_resnetNC_BNse_L",
        "epoch_crnn_tresnetE_S",
        "epoch_crnn_tresnetE_M",
        "epoch_crnn_tresnetE_L",
    ]:
        getattr(model_cfg, name).caisr_feat_dim = crnn_dim


sync_feature_config()
