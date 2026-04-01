"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG

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
# Each sample is a full-night sequence of ~720–1100 thirty-second epochs.
# batch_size=16 fits comfortably on the challenge GPUs (A30 24 GB / RTX 6000 Ada 48 GB)
# and on a local 16 GB GPU.
TrainCfg.batch_size = 16
TrainCfg.train_ratio = 0.8
TrainCfg.model_name = "epoch_transformer"  # primary model for this challenge

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

# sig_len kept for backwards-compat with raw-signal models
TrainCfg.sig_len = 3000  # 30 seconds at 100Hz

# Optimization Configs.
# 624 training records / batch_size=16 ≈ 39 steps/epoch.
# 100 epochs × 39 steps ≈ 3 900 total gradient steps — sufficient for an
# 825K-parameter Transformer trained from scratch on this dataset size.
TrainCfg.n_epochs = 100
TrainCfg.optimizer = "adamw_amsgrad"
TrainCfg.decay = 1e-2  # AdamW weight decay (standard Transformer practice)
TrainCfg.lr_scheduler = "one_cycle"
TrainCfg.max_lr = 1e-3  # OneCycleLR peak ≈ 3× base lr
TrainCfg.betas = (0.9, 0.999)
TrainCfg.grad_clip = 1.0  # gradient clipping max norm for Transformer stability

# Preprocessing
TrainCfg.normalize = CFG(
    method="z-score",
    mean=0.0,
    std=1.0,
)

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

# EpochTransformer: operates on a sequence of 30s-epoch CAISR feature vectors.
# Input: (B, T, caisr_feat_dim) + demographics (B, demographic_dim)
# Output: binary classification (cognitive impairment)
ModelCfg.epoch_transformer = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.epoch_transformer.caisr_feat_dim = 21  # CAISR_EPOCH_DIM from const.py
ModelCfg.epoch_transformer.demographic_dim = 3  # DEMOGRAPHIC_DIM from const.py
ModelCfg.epoch_transformer.d_model = 128
ModelCfg.epoch_transformer.nhead = 4
ModelCfg.epoch_transformer.num_layers = 4
ModelCfg.epoch_transformer.dim_feedforward = 512
ModelCfg.epoch_transformer.dropout = 0.1
ModelCfg.epoch_transformer.activation = "gelu"
ModelCfg.epoch_transformer.criterion = "BCEWithLogitsLoss"
ModelCfg.epoch_transformer.dem_encoder = CFG(
    enable=True,
    input_dim=3,  # Age, Sex, BMI
    hidden_dim=64,
    mode="film",  # FiLM conditioning on demographics
)

# ChannelTransformer configuration (raw-signal fallback)
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
