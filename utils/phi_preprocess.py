"""Memory-efficient replacement for the Philosopher's Stone wavelet stage.

The upstream ``philosopher_utils._compute_wavelet_specs`` runs a synchrosqueezed
CWT over the whole night at 100 Hz (~60-70 GB peak RAM, single-threaded), even
though the model only consumes the spectrogram at 1 Hz time resolution and only
uses the plain CWT magnitude (``Wx``) — the synchrosqueezed ``Tx`` is discarded.

This module reproduces ``Wx`` with a hybrid scale split:

* **Low frequencies** (``<= low_freq_cut``): the corresponding wavelets have
  support of minutes-to-hours, so their rows are computed on the FULL signal
  with the same non-vectorised ``cwt`` call the upstream path uses (bit-for-bit
  identical).  Memory: ``n_low x N x 16B`` (~10 GB instead of ~60 GB).
* **High frequencies** (``> low_freq_cut``): the wavelets are localised, so
  their rows are computed on overlapping chunks (vectorised, a few GB each) and
  stitched.  The overlap covers the largest wavelet support in the high group.

The scale grid depends on the full signal length (ssqueezepy
``process_scales``), so it is computed once for the full night and passed
explicitly to every call — this keeps the scale/frequency axes identical to the
upstream single-pass run.  The assembled ``(time, freq)`` spectrogram then goes
through the same interpolation and padding as upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHI_SRC = PROJECT_ROOT / "third_party" / "philosophers-stone" / "src"

DEFAULT_CHUNK_SECONDS = 2400.0  # 40 min per chunk (high-frequency part)
DEFAULT_OVERLAP_SECONDS = 600.0  # dropped from both sides of each chunk boundary
DEFAULT_LOW_FREQ_CUT = 2.0  # Hz; at/below this the wavelets are global


def _ensure_phi_importable() -> None:
    if str(_PHI_SRC) not in sys.path:
        sys.path.insert(0, str(_PHI_SRC))


def compute_wavelet_spectrogram_chunked(
    signal_100: np.ndarray,
    cfg,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    low_freq_cut: float = DEFAULT_LOW_FREQ_CUT,
) -> np.ndarray:
    """Return the canonical ``(hours_pad * 3600 * fs_time, n_freqs)`` spectrogram.

    ``signal_100`` must be the bandpass-filtered EEG already resampled to
    100 Hz — the same input the upstream wavelet stage receives.
    """
    _ensure_phi_importable()
    from philosophers_stone.philosopher_utils import _make_frequency_grid
    from philosophers_stone.preprocessing_and_spectrograms import (
        interpolate_wx_2d,
        pad_spectrogram,
    )
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.ssqueezing import _compute_associated_frequencies
    from ssqueezepy.utils.cwt_utils import process_scales

    fs = 100.0
    nv = int(cfg.nv)
    x = np.asarray(signal_100, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty signal")

    wavelet = Wavelet(
        (cfg.wavelet_name, {"gamma": cfg.wavelet_gamma, "beta": cfg.wavelet_beta}),
        N=int(4 * fs),
    )
    # Scale grid and frequency bins for the FULL signal length — identical to
    # the upstream single-pass run (both depend on the signal length).
    scales = process_scales("log-piecewise", n, wavelet, nv=nv)
    ssq_freqs = _compute_associated_frequencies(
        scales,
        n,
        wavelet,
        "log-piecewise",
        maprange="peak",
        was_padded=True,
        dt=1.0 / fs,
        transform="cwt",
    )

    n_low = int(np.searchsorted(ssq_freqs, low_freq_cut, side="right"))
    n_low = max(0, min(n_low, len(scales)))
    scales_high = scales[: len(scales) - n_low]
    scales_low = scales[len(scales) - n_low :]

    cwt_kwargs = dict(l1_norm=True, padtype="reflect", fs=fs)

    # --- low-frequency rows: full signal, non-vectorised (== upstream) -------
    if len(scales_low):
        wx_low, _ = cwt(x, wavelet, scales=scales_low, vectorized=False, **cwt_kwargs)
        wx_low = np.abs(wx_low)
    else:
        wx_low = np.empty((0, n), dtype=float)

    # --- high-frequency rows: overlapping chunks, vectorised ----------------
    chunk = int(round(chunk_seconds * fs))
    overlap = int(round(overlap_seconds * fs))
    if len(scales_high):
        if overlap <= 0 or chunk <= 0 or 2 * overlap >= chunk:
            raise ValueError("require chunk_seconds > 2 * overlap_seconds > 0")
        parts: list[np.ndarray] = []
        t = 0
        while t < n:
            t_end = min(t + chunk, n)
            t0 = max(0, t - overlap)
            t1 = min(n, t_end + overlap)
            seg = x[t0:t1]
            wx_high, _ = cwt(seg, wavelet, scales=scales_high, vectorized=True, **cwt_kwargs)
            wx_high = np.abs(wx_high)
            left = t - t0
            right = t1 - t_end
            parts.append(wx_high[:, left : len(seg) - right if right else None])
            t = t_end
        wx_high = np.concatenate(parts, axis=1)
    else:
        wx_high = np.empty((0, n), dtype=float)

    # columns in descending-scale order (lowest frequency first), matching the
    # upstream orientation: |Wx|[::-1].T with rows originally in ascending-scale
    # order — i.e. largest scales (low freq) first.
    wx = np.vstack([wx_low[::-1], wx_high[::-1]])
    specs_raw = wx.T  # (time, freq), matching upstream orientation
    assert specs_raw.shape[0] == n, (specs_raw.shape[0], n)

    # kill NaNs — mirror upstream
    nan_frac = np.isnan(specs_raw).mean()
    if nan_frac > 0.1:
        print("Warning: NaN fraction in spectrogram is > 10%. Unusual, check EEG and spectrogram.")
    specs_raw[np.isnan(specs_raw)] = 0

    freq_bins = _make_frequency_grid(cfg.n_freqs)
    specs_interp = interpolate_wx_2d(specs_raw, ssq_freqs, freq_bins, fs, cfg.fs_time)
    specs = pad_spectrogram(specs_interp, cfg.fs_time, hours_pad=cfg.hours_pad)
    return specs
