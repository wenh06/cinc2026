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
TrainCfg.batch_size = 32
TrainCfg.train_ratio = 0.8
TrainCfg.sig_len = 3000  # 30 seconds at 100Hz
TrainCfg.model_name = "transformer"  # Default model choice: "transformer" or "multibranch"

# Optimization Configs
TrainCfg.n_epochs = 50
TrainCfg.lr = 3e-4
TrainCfg.optimizer = "adamw_amsgrad"
TrainCfg.decay = 1e-2
TrainCfg.lr_scheduler = "one_cycle"
TrainCfg.max_lr = 1e-3
TrainCfg.betas = (0.9, 0.999)

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
    patience=15,
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

# ChannelTransformer configuration
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

# MultiBranchNet configuration
ModelCfg.multibranch = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.multibranch.d_model = 128
ModelCfg.multibranch.modalities = ["eeg", "eog", "emg", "ecg", "resp"]
ModelCfg.multibranch.nhead = 4
ModelCfg.multibranch.dropout = 0.1
ModelCfg.multibranch.criterion = "CrossEntropyLoss"
ModelCfg.multibranch.dem_encoder = deepcopy(ModelCfg.transformer.dem_encoder)

# adjust filter lengths if needed
cnn_filter_length_ratio = 1.0
