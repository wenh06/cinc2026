"""Epoch-Transformer model configuration building blocks for CinC 2026.

:data:`EPOCH_TRANSFORMER_BASE` contains the hyper-parameters that are
shared across all size presets (S / M / L).  Size-specific values
(``d_model``, ``nhead``, ``num_layers``, ``dim_feedforward``) are set
per-preset in :mod:`cfg`.

Size summary
------------
=========  =======  =====  ==========  ===========  ~params
Preset     d_model  nhead  num_layers  dim_ff       -------
=========  =======  =====  ==========  ===========  ~params
``_S``     64       2      2           256          ~110 K
``_M``     128      4      4           512          ~825 K  *(default)*
``_L``     256      8      6           1024         ~5.3 M
=========  =======  =====  ==========  ===========  ~params
"""

from torch_ecg.cfg import CFG

__all__ = ["EPOCH_TRANSFORMER_BASE"]

# Shared config (size-independent parameters)
EPOCH_TRANSFORMER_BASE = CFG()
EPOCH_TRANSFORMER_BASE.caisr_feat_dim = 21  # CAISR_EPOCH_DIM
EPOCH_TRANSFORMER_BASE.demographic_dim = 3  # age/100, sex(0/1), bmi/50
EPOCH_TRANSFORMER_BASE.dropout = 0.1
EPOCH_TRANSFORMER_BASE.activation = "gelu"
EPOCH_TRANSFORMER_BASE.criterion = "BCEWithLogitsLoss"
EPOCH_TRANSFORMER_BASE.dem_encoder = CFG(
    enable=True,
    input_dim=3,
    hidden_dim=64,
    mode="film",
)
