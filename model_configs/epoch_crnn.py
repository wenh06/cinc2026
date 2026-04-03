"""Epoch-CRNN model configuration building blocks for CinC 2026.

Defines :data:`EPOCH_CRNN_CONFIG` — a base config in the
:class:`~torch_ecg.model_configs.ECG_CRNN_CONFIG` style — with several
epoch-scale ResNet backbone variants pre-registered.  Size presets
(S / M / L) are assembled in :mod:`cfg` by selecting a ``cnn.name`` and
overriding ``rnn.lstm.hidden_sizes`` and ``clf.out_channels``.

Available CNN backbones
-----------------------
All variants are lightweight ResNets with small epoch-scale kernels.
"Epoch-scale" means kernel sizes are counted in *30-second epochs*, not in
raw signal samples; there is no sampling-frequency concept here.

**resnetN family** — 3-stage ResNet-N (basic blocks):

========================  =====================  ==================
Config key                Channel widths         Comment
========================  =====================  ==================
``resnetN_S``             16 → 32 → 64           Small
``resnetN_M`` *(default)* 32 → 64 → 128          Medium
``resnetN_L``             64 → 128 → 256         Large
``resnetNS_M``            32 → 64 → 128          Separable-conv M
``resnetNB_M``            32 → 64 → 128          Bottleneck-block M
========================  =====================  ==================

**resnetNC_BNse family** — 4-stage Nature-Comm style with bottleneck+SE blocks.
Inspired by ``resnet_nature_comm_bottle_neck_se`` from torch_ecg, adapted for
epoch sequences (small kernels, stride-2 instead of stride-4, no max-pool stem).
The bottleneck expansion factor is 4, so actual CNN output channels =
``num_filters[-1] * 4``:

=========================  ===================  ========================
Config key                 ``num_filters``      CNN output channels
=========================  ===================  ========================
``resnetNC_BNse_S``        16 → 32 → 48 → 64   256 (64 × 4)
``resnetNC_BNse_M``        32 → 64 → 96 → 128  512 (128 × 4)
``resnetNC_BNse_L``        48 → 96 → 192 → 256 1024 (256 × 4)
=========================  ===================  ========================

**tresnetE family** — Epoch-scale TResNet.
Mixed-block structure [basic_se, basic_se, bottleneck_se, bottleneck_B] from
``tresnetF``, adapted for epoch sequences: standard conv stem (no s2d), small
kernels, max subsampling (no blur), 4 stages of stride-2.  CNN output channels
are large due to bottleneck expansion on the last two stages:

=================  ==========================  ========================
Config key         ``num_filters``             CNN output channels
=================  ==========================  ========================
``tresnetE_S``     16 → 32 → 64 → 128         512  (128 × 4)
``tresnetE_M``     32 → 64 → 128 → 256        1024 (256 × 4)
``tresnetE_L``     48 → 96 → 192 → 384        1536 (384 × 4)
=================  ==========================  ========================

Available RNN choices (``rnn.name``)
-------------------------------------
``"lstm"`` (default)
    Bidirectional LSTM with ``retseq=False``.  Output shape: ``(B, 2·hidden)``.
``"none"``
    Skip the RNN; the CNN output is globally max-pooled instead.

Attention (``attn.name``) and global pool
------------------------------------------
Attention is ``"none"`` by default — the BiLSTM already aggregates the
temporal sequence.  ``global_pool = "none"`` because ``retseq=False``
already produces a fixed-size vector.
"""

from copy import deepcopy

from torch_ecg.cfg import CFG
from torch_ecg.model_configs.cnn.resnet import (
    resnet_block_basic,
    resnet_bottle_neck_B,
    resnet_bottle_neck_se,
    resnetN,
    tresnetF,
)
from torch_ecg.model_configs.rnn import lstm

__all__ = ["EPOCH_CRNN_CONFIG"]

# ─────────────────────────────────────────────────────────────────────────────
# Shared stem / block settings for all epoch-scale ResNet variants
# ─────────────────────────────────────────────────────────────────────────────

_EPOCH_STEM = CFG()
_EPOCH_STEM.filter_lengths = 5  # 5-epoch context in stem  (≈ 2.5 min)
_EPOCH_STEM.conv_stride = 1  # no stride in stem
_EPOCH_STEM.pool_size = 0  # disable MaxPool — avoid over-reducing T
_EPOCH_STEM.pool_stride = 0

_EPOCH_BLOCK = deepcopy(resnet_block_basic)
_EPOCH_BLOCK.activation = "relu"
_EPOCH_BLOCK.kw_activation = {"inplace": True}


def _make_resnetN_epoch(num_filters: list, building_block: str = "basic") -> CFG:
    """Build an epoch-scale ResNet-N config.

    Parameters
    ----------
    num_filters : list of int, length 3
        Output channel counts for the three backbone stages.  The stem
        uses ``num_filters[0]`` channels.
    building_block : {"basic", "bottleneck"}
        Which residual block type to use.
    """
    rn = deepcopy(resnetN)
    rn.num_blocks = [1, 1, 1]  # three stages, one block each
    rn.num_filters = list(num_filters)
    rn.filter_lengths = [5, 3, 3]  # k=5 (150 s context), k=3 (90 s)
    rn.subsample_lengths = [2, 2, 2]  # stride 2 per stage → T/8 total

    stem = _EPOCH_STEM
    rn.stem = deepcopy(stem)
    rn.stem.num_filters = num_filters[0]

    if building_block == "bottleneck":
        rn.building_block = "bottleneck"
        rn.block = deepcopy(resnet_bottle_neck_B)
        rn.block.activation = "relu"
        rn.block.kw_activation = {"inplace": True}
    else:
        rn.building_block = "basic"
        rn.block = deepcopy(_EPOCH_BLOCK)

    return rn


def _make_resnetNS_epoch(num_filters: list) -> CFG:
    """Build an epoch-scale ResNet-N config with **separable** convolutions."""
    rn = _make_resnetN_epoch(num_filters, building_block="basic")
    rn.block.conv_type = "separable"
    return rn


# ── New family: resnetNC_BNse — Nature-Comm style 4-stage with Bottleneck + SE ──
#
# Inspired by ``resnet_nature_comm_bottle_neck_se`` from torch_ecg.
# Key adaptations for epoch sequences:
#   • filter_lengths [5,3,3,3]  (was 17)
#   • subsample_lengths [2,2,2,2]  (was [4,4,4,4])
#   • stem: conv_stride=1, no MaxPool  (was stride-2 + pool)
#   • activation "relu"  (unchanged)
#   • SE squeeze-excitation on every bottleneck block
# CNN output channels = num_filters[-1] * expansion (expansion=4 for bottleneck).


def _make_resnetNC_BNse_epoch(num_filters: list) -> CFG:
    """Build a 4-stage epoch-scale bottleneck+SE ResNet config.

    Parameters
    ----------
    num_filters : list of 4 ints
        Bottleneck-width (pre-expansion) for each of the 4 stages.
        Final CNN output channels = ``num_filters[-1] * 4``.
    """
    rn = deepcopy(resnetN)
    rn.num_blocks = [1, 1, 1, 1]
    rn.building_block = "bottleneck"
    rn.block = deepcopy(resnet_bottle_neck_se)
    rn.block.activation = "relu"
    rn.block.kw_activation = {"inplace": True}
    rn.block.subsample_mode = "max"
    rn.num_filters = list(num_filters)
    rn.filter_lengths = [5, 3, 3, 3]
    rn.subsample_lengths = [2, 2, 2, 2]
    rn.stem = deepcopy(_EPOCH_STEM)
    rn.stem.num_filters = num_filters[0]
    rn.dropouts = 0.1
    return rn


# ── New family: tresnetE — Epoch-scale TResNet ────────────────────────────────
#
# Adapted from ``tresnetF`` (the femto variant of TResNet):
# mixed-block structure [basic_se, basic_se, bottleneck_se, bottleneck_B].
# Key adaptations:
#   • stem subsample_mode "max"  (was "s2d" — space-to-depth, not applicable here)
#   • stem filter_lengths 5, conv_stride 1, no MaxPool
#   • filter_lengths [5,3,3,3]  (was 17)
#   • subsample_lengths [2,2,2,2]  (was [1,2,2,2])
#   • conv_type removed (was "separable"/"aa")
#   • subsample_mode "max" for every block  (was "blur")
#   • activation "relu"  (was "leaky")
# CNN output channels = num_filters[-1] * 4 (last two stages are bottleneck).


def _make_tresnetE_epoch(num_filters: list) -> CFG:
    """Build a 4-stage epoch-scale TResNet config.

    Block structure: [BasicSE, BasicSE, BottleneckSE, BottleneckB].
    This mirrors the TResNet mixed-block design, adapted for epoch-scale
    sequences (no raw-signal specific tricks).

    Parameters
    ----------
    num_filters : list of 4 ints
        Channel widths for each stage (bottleneck width for stages 3-4).
        Final CNN output channels = ``num_filters[-1] * 4``.
    """
    cfg = deepcopy(tresnetF)
    cfg.stem.subsample_mode = "max"  # replace space-to-depth
    cfg.stem.filter_lengths = 5
    cfg.stem.conv_stride = 1
    cfg.stem.pool_size = 0
    cfg.stem.pool_stride = 0
    cfg.stem.num_filters = num_filters[0]
    cfg.filter_lengths = [5, 3, 3, 3]
    cfg.subsample_lengths = [2, 2, 2, 2]
    cfg.num_filters = list(num_filters)
    cfg.dropouts = 0.1
    cfg.activation = "relu"
    cfg.kw_activation = {"inplace": True}
    for b in cfg.block:
        b.pop("conv_type", None)  # remove aa/separable conv
        b.subsample_mode = "max"  # replace blur
        b.activation = "relu"
        b.kw_activation = {"inplace": True}
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Base config — collects all available backbones + default RNN/attn/clf
# ─────────────────────────────────────────────────────────────────────────────

EPOCH_CRNN_CONFIG = CFG()

# ── CNN: select one of the registered keys below ─────────────────────────────
EPOCH_CRNN_CONFIG.cnn = CFG()
EPOCH_CRNN_CONFIG.cnn.name = "resnetN_M"  # default

EPOCH_CRNN_CONFIG.cnn.resnetN_S = _make_resnetN_epoch([16, 32, 64])
EPOCH_CRNN_CONFIG.cnn.resnetN_M = _make_resnetN_epoch([32, 64, 128])
EPOCH_CRNN_CONFIG.cnn.resnetN_L = _make_resnetN_epoch([64, 128, 256])
EPOCH_CRNN_CONFIG.cnn.resnetNS_M = _make_resnetNS_epoch([32, 64, 128])
EPOCH_CRNN_CONFIG.cnn.resnetNB_M = _make_resnetN_epoch([32, 64, 128], building_block="bottleneck")

# 4-stage bottleneck+SE backbone (Nature-Comm style, epoch-adapted)
# CNN output channels = num_filters[-1] * 4 (bottleneck expansion)
EPOCH_CRNN_CONFIG.cnn.resnetNC_BNse_S = _make_resnetNC_BNse_epoch([16, 32, 48, 64])
EPOCH_CRNN_CONFIG.cnn.resnetNC_BNse_M = _make_resnetNC_BNse_epoch([32, 64, 96, 128])
EPOCH_CRNN_CONFIG.cnn.resnetNC_BNse_L = _make_resnetNC_BNse_epoch([48, 96, 192, 256])

# 4-stage TResNet-style backbone (epoch-adapted, mixed basic+bottleneck+SE)
# CNN output channels = num_filters[-1] * 4 (last two stages are bottleneck)
EPOCH_CRNN_CONFIG.cnn.tresnetE_S = _make_tresnetE_epoch([16, 32, 64, 128])
EPOCH_CRNN_CONFIG.cnn.tresnetE_M = _make_tresnetE_epoch([32, 64, 128, 256])
EPOCH_CRNN_CONFIG.cnn.tresnetE_L = _make_tresnetE_epoch([48, 96, 192, 384])

# ── RNN ───────────────────────────────────────────────────────────────────────
EPOCH_CRNN_CONFIG.rnn = CFG()
EPOCH_CRNN_CONFIG.rnn.name = "lstm"

# Copy the torch_ecg default lstm config then customise for our task
_lstm_base = deepcopy(lstm)
_lstm_base.hidden_sizes = [128]  # per-direction hidden size (overridden per preset)
_lstm_base.bidirectional = True
_lstm_base.retseq = False  # last hidden → (B, 2·hidden); no per-step output
_lstm_base.dropouts = 0.1
_lstm_base.bias = True
EPOCH_CRNN_CONFIG.rnn.lstm = _lstm_base

# ── Attention ─────────────────────────────────────────────────────────────────
EPOCH_CRNN_CONFIG.attn = CFG()
EPOCH_CRNN_CONFIG.attn.name = "none"  # BiLSTM already aggregates; no extra attn

# ── Global pool ───────────────────────────────────────────────────────────────
# retseq=False already yields a fixed (B, 2·hidden) vector; skip global pool.
EPOCH_CRNN_CONFIG.global_pool = "none"
EPOCH_CRNN_CONFIG.global_pool_size = 1

# ── Intermediate MLP head (final output layer is added by EpochCRNN) ─────────
# out_channels = intermediate hidden sizes before the final 1-logit output.
# This will be replaced inside EpochCRNN.__init__; we keep it here so that
# the ECG_CRNN parent init does not fail.
EPOCH_CRNN_CONFIG.clf = CFG(
    out_channels=[64],
    activation="gelu",
    bias=True,
    dropouts=0.1,
    kernel_initializer="he_normal",
)
