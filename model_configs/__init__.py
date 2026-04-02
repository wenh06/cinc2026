"""Model configuration building blocks for CinC 2026.

Mirrors the torch_ecg model_configs pattern: each sub-module defines a
*base* config dict that registers all available backbone/component variants.
Size presets (S / M / L) are assembled in :mod:`cfg` from these base configs.
"""

from .epoch_crnn import EPOCH_CRNN_CONFIG
from .epoch_transformer import EPOCH_TRANSFORMER_BASE

__all__ = [
    "EPOCH_CRNN_CONFIG",
    "EPOCH_TRANSFORMER_BASE",
]
