"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.model_configs import ECG_CRNN_CONFIG, linear  # noqa: F401
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths  # noqa: F401

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

BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

TrainCfg.checkpoints = BaseCfg.checkpoints
TrainCfg.checkpoints.mkdir(exist_ok=True)


###############################################################################
# configurations for building deep learning models
###############################################################################

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype

# _BASE_MODEL_CONFIG.criterion = TrainCfg.criterion
# _BASE_MODEL_CONFIG.criterion_kw = TrainCfg.criterion_kw.copy()


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

# Model configuration for ChannelTransformer
ModelCfg.transformer = CFG()
ModelCfg.transformer.d_model = 128
ModelCfg.transformer.nhead = 4
ModelCfg.transformer.num_layers = 4
ModelCfg.transformer.dim_feedforward = 512
ModelCfg.transformer.dropout = 0.1
ModelCfg.transformer.activation = "relu"
ModelCfg.transformer.max_channels = 25  # Max unique channels across all recordings
ModelCfg.transformer.classes = ["Negative", "Positive"]
ModelCfg.transformer.num_classes = len(ModelCfg.transformer.classes)

# Demographic Encoder configuration
ModelCfg.transformer.dem_encoder = CFG()
ModelCfg.transformer.dem_encoder.enable = True
ModelCfg.transformer.dem_encoder.input_dim = 5  # e.g., Age, Sex, BMI, etc.
ModelCfg.transformer.dem_encoder.hidden_dim = 64
ModelCfg.transformer.dem_encoder.mode = "film"  # or "concat"

# Loss configuration
ModelCfg.transformer.criterion = "CrossEntropyLoss"
ModelCfg.transformer.criterion_kw = CFG()

# Model configuration for MultiBranchNet
ModelCfg.multibranch = CFG()
ModelCfg.multibranch.d_model = 128
ModelCfg.multibranch.modalities = ["eeg", "eog", "emg", "ecg", "resp"]
ModelCfg.multibranch.classes = ModelCfg.transformer.classes
ModelCfg.multibranch.num_classes = len(ModelCfg.multibranch.classes)
ModelCfg.multibranch.dem_encoder = deepcopy(ModelCfg.transformer.dem_encoder)
ModelCfg.multibranch.criterion = ModelCfg.transformer.criterion
ModelCfg.multibranch.criterion_kw = ModelCfg.transformer.criterion_kw.copy()

# adjust filter lengths, > 1 for enlarging, < 1 for shrinking
cnn_filter_length_ratio = 1.0
