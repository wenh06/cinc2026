"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG

from model_configs import EPOCH_CRNN_CONFIG, EPOCH_TRANSFORMER_BASE

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
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
# batch_size=16: 624 train records / 16 ≈ 39 steps/epoch — enough gradient noise for
# regularisation and sufficient steps for OneCycleLR to anneal properly.
# (batch_size=64 collapses to ~10 steps/epoch which hurt CRNN convergence in run 04-04.)
TrainCfg.batch_size = 16
# train_ratio: fallback 80/20 split used by CINC2026Dataset when the canonical
# JSON split file (utils/cinc2026-data-split.json) is absent.
TrainCfg.train_ratio = 0.8
TrainCfg.model_name = "epoch_crnn_M"  # current model for submission 3

# learning_rate is the canonical name used by BaseTrainer; lr is kept as an alias
TrainCfg.lr = 3e-4
TrainCfg.learning_rate = TrainCfg.lr

# db_dir must be set at training time (e.g. via command-line argument)
TrainCfg.db_dir = None

# Monitor metric for model selection and early stopping
TrainCfg.monitor = "auroc"

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

# sig_len: raw-signal legacy kept for backwards-compat with models/transformer.py
# prototype.  Not used in the CAISR epoch-feature pipeline.
TrainCfg.sig_len = 3000  # 30 seconds at 100Hz

# Optimization Configs.
# 624 train records / batch_size=16 ≈ 39 steps/epoch.
# 100 epochs × 39 ≈ 3 900 total gradient steps.
TrainCfg.n_epochs = 100
TrainCfg.optimizer = "adamw_amsgrad"
# weight_decay: BaseTrainer._setup_optimizer reads get_kwargs(AdamW) which uses the
# key "weight_decay" — NOT "decay".  Always use "weight_decay" here.
TrainCfg.weight_decay = 1e-2
TrainCfg.lr_scheduler = "one_cycle"
TrainCfg.max_lr = 1e-3  # OneCycleLR peak
# pct_start: fraction of total steps used for LR warm-up (OneCycleLR).
# 0.3 (default) is good for Transformer; use 0.1 for CRNN which converges faster.
TrainCfg.pct_start = 0.3
TrainCfg.betas = (0.9, 0.999)
TrainCfg.grad_clip = 1.0  # gradient clipping max norm (0 to disable)

# Augmentation / Regularisation
# CAISR epoch features are pre-bounded in [0,1] by construction (one-hot stages,
# softmax probs, event fractions, sin/cos position encoding, scaled demographics).
# No z-score or amplitude normalisation is needed or appropriate here.
# PreprocManager (BandPass / ZScoreNormalize / Resample) is designed for raw signals
# and should only be wired in if a raw-signal model branch is added in the future.
#
# label_smoothing: smooths targets {0,1} → {ε/2, 1-ε/2} to discourage over-confident
# predictions and improve calibration when test prevalence differs from training.
TrainCfg.label_smoothing = 0.05

# pos_weight: BCEWithLogitsLoss pos_weight.
# Hidden test set appears to have ~6% positive prevalence vs 50% in training.
# Set to None to disable (balanced training, currently preferred for stability).
TrainCfg.pos_weight = None  # e.g. 5.0 to upweight positives

# Preprocessing — commented out: CAISR features need no additional normalisation.
# Uncomment and wire into a PreprocManager only if raw physiological signals are used.
# TrainCfg.normalize = CFG(
#     method="z-score",
#     mean=0.0,
#     std=1.0,
# )

# Callbacks & Logging
TrainCfg.log_step = 20
TrainCfg.keep_checkpoint_max = 5
TrainCfg.early_stopping = CFG(
    min_delta=0.001,
    patience=20,  # with 100 epochs; stops ~20 epochs after last improvement
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
