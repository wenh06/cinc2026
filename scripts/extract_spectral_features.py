#!/usr/bin/env python
"""Extract Ye-2023-style tabular features from raw PSG + CAISR annotations.

D1 pipeline.  Feature families (all computable at inference from EDF + CAISR):

1. Per-stage relative band powers (delta/theta/alpha/sigma/beta) per bipolar
   EEG derivation, plus theta/alpha ratio, delta/alpha ratio, delta/beta ratio,
   Hjorth activity/mobility/complexity, clipped kurtosis, spectral edge 95.
2. Temporal quantile pooling of per-epoch band powers over scored sleep:
   q10/25/50/75/88/95, IQR, range80, CV, first-vs-last-third drift, worst-hour
   concentration, excursion rate (I-CARE 2023 winner methodology).
3. Inter-channel magnitude-squared coherence per channel pair, stage and band
   (scale-invariant -> robust to per-channel gain/reference differences).
4. HRV from ECG (SDNN, RMSSD, pNN50, mean HR) whole-night / NREM / REM.
5. SpO2 (min, mean, ODI4, fraction below 88%, hypoxic burden).
6. CAISR sleep architecture (stage fractions, 5x5 transitions, W/REM bout
   stats, awakenings, N3 half-night difference, arousal/AHI/PLM indices,
   stage-prob entropy).
7. Metadata (age, sex, BMI, recording year, site, label for analysis only).

Deterministic: no RNG anywhere.  CPU-only.  Uses ``helper_code.py`` for channel
standardization; reads only the needed channels per record to bound memory.

Usage:
    python scripts/extract_spectral_features.py \
        --data-root /Data1/wenh06/physionetchallenge2026data \
        --out-dir tmp/spectral_features \
        --limit 10                # smoke run
    python scripts/extract_spectral_features.py \
        --data-root /Data1/wenh06/physionetchallenge2026data \
        --out-dir tmp/spectral_features --workers 8
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import coherence, find_peaks, welch
from scipy.stats import entropy as scipy_entropy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

import helper_code

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS = OrderedDict(
    [
        ("delta", (0.5, 4.0)),
        ("theta", (4.0, 8.0)),
        ("alpha", (8.0, 12.0)),
        ("sigma", (12.0, 16.0)),
        ("beta", (16.0, 30.0)),
    ]
)
TOTAL_BAND = (0.5, 30.0)
EPOCH_SECONDS = 30.0

# CAISR stage encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=unscored.
STAGES = OrderedDict([("wake", 5), ("n1", 3), ("n2", 2), ("n3", 1), ("rem", 4)])
SLEEP_STAGE_VALUES = (1, 2, 3, 4)

EEG_TARGETS = ["c3-m2", "c4-m1", "f3-m2", "f4-m1", "o1-m2", "o2-m1"]
# Monopolar derivation targets: target -> (active, reference).
MONO_PAIRS = {
    "c3-m2": ("c3", "m2"),
    "c4-m1": ("c4", "m1"),
    "f3-m2": ("f3", "m2"),
    "f4-m1": ("f4", "m1"),
    "o1-m2": ("o1", "m2"),
    "o2-m1": ("o2", "m1"),
}
COHERENCE_PAIRS = [
    ("c3-m2", "c4-m1"),
    ("f3-m2", "f4-m1"),
    ("o1-m2", "o2-m1"),
    ("f3-m2", "c3-m2"),
    ("f4-m1", "c4-m1"),
]
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.88, 0.95)

MAX_STAGE_CONCAT_SECONDS = 1800.0  # cap stage-concatenated segments (30 min)
MAX_COHERENCE_SECONDS = 1800.0


# ---------------------------------------------------------------------------
# Feature templates (stable column order)
# ---------------------------------------------------------------------------


def _stage_spectral_template() -> OrderedDict:
    """Per-stage x per-band x per-channel relative power + derived scalars."""
    feat = OrderedDict()
    for stage in STAGES:
        for band in list(BANDS) + ["theta_alpha", "delta_alpha", "delta_beta"]:
            for ch in EEG_TARGETS:
                feat[f"spec_{stage}_{band}_{ch}"] = np.nan
    for stage in STAGES:
        for ch in EEG_TARGETS:
            for scalar in ["kurtosis", "hj_activity", "hj_mobility", "hj_complexity", "edge95"]:
                feat[f"spec_{stage}_{scalar}_{ch}"] = np.nan
    return feat


def _temporal_template() -> OrderedDict:
    feat = OrderedDict()
    for band in BANDS:
        for q in QUANTILES:
            feat[f"tp_{band}_q{int(round(q * 100))}"] = np.nan
        for scalar in ["iqr", "range80", "cv", "drift", "worst_hour_frac", "excursion_rate"]:
            feat[f"tp_{band}_{scalar}"] = np.nan
    for ratio in ["delta_alpha", "theta_alpha", "delta_beta"]:
        feat[f"tp_ratio_{ratio}_q88"] = np.nan
    return feat


def _coherence_template() -> OrderedDict:
    feat = OrderedDict()
    for stage in STAGES:
        for a, b in COHERENCE_PAIRS:
            for band in BANDS:
                feat[f"coh_{a.replace('-', '')}_{b.replace('-', '')}_{band}_{stage}"] = np.nan
    return feat


def _hrv_template() -> OrderedDict:
    feat = OrderedDict()
    for context in ["all", "nrem", "rem"]:
        for scalar in ["sdnn_ms", "rmssd_ms", "pnn50", "mean_hr_bpm"]:
            feat[f"hrv_{scalar}_{context}"] = np.nan
    return feat


def _spo2_template() -> OrderedDict:
    return OrderedDict(
        [
            ("spo2_min", np.nan),
            ("spo2_mean", np.nan),
            ("spo2_odi4", np.nan),
            ("spo2_pct_below88", np.nan),
            ("spo2_hypoxic_burden", np.nan),
        ]
    )


def _arch_template() -> OrderedDict:
    feat = OrderedDict()
    for stage in STAGES:
        feat[f"arch_frac_{stage}"] = np.nan
    for src in STAGES:
        for dst in STAGES:
            feat[f"trans_{src}_{dst}"] = np.nan
    for scalar in [
        "n_epochs",
        "total_sleep_time_h",
        "sleep_efficiency",
        "n_awakenings",
        "bout_mean_W",
        "bout_std_W",
        "bout_mean_R",
        "bout_std_R",
        "n_rem_bouts",
        "n3_first_minus_second_frac",
        "stage_entropy",
        "stage_prob_entropy_mean",
        "stage_prob_entropy_std",
        "arousal_index",
        "ahi",
        "plmi",
    ]:
        feat[f"arch_{scalar}"] = np.nan
    return feat


def _meta_template() -> OrderedDict:
    return OrderedDict(
        [
            ("meta_age", np.nan),
            ("meta_sex_male", np.nan),
            ("meta_bmi", np.nan),
            ("meta_rec_year", np.nan),
            ("meta_site", None),
            ("label", None),
        ]
    )


FEATURE_TEMPLATE = OrderedDict()
for tpl in (
    _stage_spectral_template(),
    _temporal_template(),
    _coherence_template(),
    _hrv_template(),
    _spo2_template(),
    _arch_template(),
):
    FEATURE_TEMPLATE.update(tpl)
META_TEMPLATE = _meta_template()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def load_caisr(ann_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load algorithmic annotation signals from a CAISR annotation EDF."""
    try:
        with pyedflib.EdfReader(ann_path) as edf:
            labels = edf.getSignalLabels()
            out = {}
            for i, label in enumerate(labels):
                out[label] = edf.readSignal(i).astype(np.float64)
        return out
    except Exception:
        return None


def read_raw_channels(edf_path: str, wanted: set) -> Dict[str, Tuple[np.ndarray, float]]:
    """Read only the raw channels in ``wanted`` (standard names)."""
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    if not wanted:
        return out
    try:
        with pyedflib.EdfReader(edf_path) as edf:
            labels = edf.getSignalLabels()
            for i, label in enumerate(labels):
                if label in wanted:
                    out[label] = (edf.readSignal(i).astype(np.float64), float(edf.getSampleFrequency(i)))
    except Exception:
        return {}
    return out


def _band_power_relative(sig: np.ndarray, fs: float) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """Relative band powers + spectral edge 95 over a single signal."""
    if len(sig) < int(4 * fs) or not np.isfinite(sig).all():
        return None, None
    nperseg = int(min(4 * fs, len(sig)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    total_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    total = np.trapezoid(psd[total_mask], freqs[total_mask])
    if total <= 0:
        return None, None
    powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        powers[name] = float(np.trapezoid(psd[mask], freqs[mask]) / total)
    # spectral edge 95 within 0.5-30 Hz
    cum = np.cumsum(psd[total_mask]) / total
    edge_idx = np.searchsorted(cum, 0.95)
    edge95 = float(freqs[total_mask][min(edge_idx, len(cum) - 1)])
    return powers, edge95


def _epoch_band_powers(sig: np.ndarray, fs: float, stages: np.ndarray) -> Optional[np.ndarray]:
    """Per-epoch relative band powers, aligned to the stage signal. (n_epochs, 5)."""
    per_epoch = int(round(EPOCH_SECONDS * fs))
    if per_epoch < 8 or len(stages) == 0:
        return None
    n_epochs = min(len(stages), len(sig) // per_epoch)
    if n_epochs < 4:
        return None
    out = np.full((n_epochs, len(BANDS)), np.nan)
    nperseg = int(min(4 * fs, per_epoch))
    for i in range(n_epochs):
        seg = sig[i * per_epoch : (i + 1) * per_epoch]
        if len(seg) < nperseg or not np.isfinite(seg).all():
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
        total_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
        total = np.trapezoid(psd[total_mask], freqs[total_mask])
        if total <= 0:
            continue
        for j, (lo, hi) in enumerate(BANDS.values()):
            mask = (freqs >= lo) & (freqs < hi)
            out[i, j] = np.trapezoid(psd[mask], freqs[mask]) / total
    return out


def _stage_concat(sig: np.ndarray, fs: float, stages: np.ndarray, stage_val: int, max_seconds: float) -> Optional[np.ndarray]:
    """Concatenate epochs scored as ``stage_val``, capped at ``max_seconds``."""
    per_epoch = int(round(EPOCH_SECONDS * fs))
    if per_epoch < 8 or len(stages) == 0:
        return None
    n_epochs = min(len(stages), len(sig) // per_epoch)
    idx = np.flatnonzero(np.asarray(stages[:n_epochs]) == stage_val)
    if idx.size == 0:
        return None
    limit = int(max_seconds * fs)
    parts, total = [], 0
    for i in idx:
        seg = sig[i * per_epoch : (i + 1) * per_epoch]
        if len(seg) < per_epoch or not np.isfinite(seg).all():
            continue
        parts.append(seg)
        total += len(seg)
        if total >= limit:
            break
    if not parts:
        return None
    out = np.concatenate(parts)
    return out[:limit]


def _hjorth(sig: np.ndarray) -> Tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    dx = np.diff(sig)
    ddx = np.diff(dx)
    activity = float(np.var(sig))
    mobility = float(np.sqrt(np.var(dx) / activity)) if activity > 0 else np.nan
    denom = np.var(dx)
    complexity = float(np.sqrt(np.var(ddx) / denom) / mobility) if denom > 0 and mobility > 0 else np.nan
    return activity, mobility, complexity


def _clipped_kurtosis(sig: np.ndarray) -> float:
    if len(sig) < 16:
        return np.nan
    p1, p99 = np.percentile(sig, [1, 99])
    x = np.clip(sig, p1, p99)
    mu = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return np.nan
    n = len(x)
    return float(((x - mu) ** 4).sum() / n / sd**4 - 3.0)


def _pair_coherence(sig_a: np.ndarray, sig_b: np.ndarray, fs: float) -> Optional[Dict[str, float]]:
    n = min(len(sig_a), len(sig_b))
    if n < int(4 * fs):
        return None
    nperseg = int(min(4 * fs, n))
    freqs, cxy = coherence(sig_a[:n], sig_b[:n], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(cxy[mask])) if mask.any() else np.nan
    return out


def _hrv_features(ecg: np.ndarray, fs: float, stage_masks: Dict[str, np.ndarray], epoch_len: int) -> Dict[str, float]:
    """R-peak-based HRV. ``stage_masks`` are boolean masks over epochs."""
    out = {f"hrv_{s}_{c}": np.nan for c in ["all", "nrem", "rem"] for s in ["sdnn_ms", "rmssd_ms", "pnn50", "mean_hr_bpm"]}
    if len(ecg) < int(30 * fs):
        return out
    min_dist = int(0.35 * fs)
    try:
        peaks, _ = find_peaks(ecg, distance=max(1, min_dist), height=np.percentile(ecg, 70))
    except Exception:
        return out
    if len(peaks) < 10:
        return out
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 250) & (rr < 2000)]  # 30-240 bpm guard
    contexts = {"all": np.ones(len(rr), dtype=bool)}
    if epoch_len > 0:
        # map each RR interval to the epoch of its first peak
        peak_epoch = (peaks / epoch_len).astype(int)
        for cname, mask in stage_masks.items():
            n = len(rr)
            ctx = np.zeros(n, dtype=bool)
            for k in range(n):
                e = peak_epoch[k] if k < len(peak_epoch) else -1
                if 0 <= e < len(mask) and mask[e]:
                    ctx[k] = True
            contexts[cname] = ctx
    for cname, ctx in contexts.items():
        r = rr[ctx]
        if len(r) < 10:
            continue
        sdnn = float(np.std(r))
        rmssd = float(np.sqrt(np.mean(np.diff(r) ** 2)))
        pnn50 = float(np.mean(np.abs(np.diff(r)) > 50))
        hr = float(60000.0 / np.mean(r))
        out[f"hrv_sdnn_ms_{cname}"] = sdnn
        out[f"hrv_rmssd_ms_{cname}"] = rmssd
        out[f"hrv_pnn50_{cname}"] = pnn50
        out[f"hrv_mean_hr_bpm_{cname}"] = hr
    return out


def _spo2_features(spo2_raw: np.ndarray, fs: float, n_epochs: int) -> Dict[str, float]:
    out = {
        "spo2_min": np.nan,
        "spo2_mean": np.nan,
        "spo2_odi4": np.nan,
        "spo2_pct_below88": np.nan,
        "spo2_hypoxic_burden": np.nan,
    }
    if len(spo2_raw) < int(60 * fs) or n_epochs < 2:
        return out
    # 30-second mean series
    per_epoch = int(round(EPOCH_SECONDS * fs))
    n = min(n_epochs, len(spo2_raw) // per_epoch)
    if n < 2:
        return out
    series = np.array([np.mean(spo2_raw[i * per_epoch : (i + 1) * per_epoch]) for i in range(n)])
    series = series[np.isfinite(series)]
    if len(series) < 2:
        return out
    # Site-dependent scale: I0002 stores fractions (0-1), S0001/I0006 store %.
    if np.nanmax(series) <= 1.5:
        series = series * 100.0
    # Disconnect/zero artifacts: keep only physiologically plausible saturations.
    series = series[(series >= 40.0) & (series <= 100.0)]
    if len(series) < 2:
        return out
    out["spo2_min"] = float(np.min(series))
    out["spo2_mean"] = float(np.mean(series))
    out["spo2_pct_below88"] = float(np.mean(series < 88))
    out["spo2_hypoxic_burden"] = float(np.mean(np.clip(90.0 - series, 0, None)))
    # ODI4: >=4% drops from a rolling 2-minute maximum baseline
    events = 0
    for i in range(2, len(series)):
        baseline = np.max(series[max(0, i - 4) : i - 1])
        if baseline - series[i] >= 4.0:
            events += 1
    hours = len(series) * EPOCH_SECONDS / 3600.0
    out["spo2_odi4"] = float(events / hours)
    return out


def _caisr_arch(ann: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Sleep-architecture features from CAISR annotations; returns (feat, stages)."""
    feat = {k: np.nan for k in _arch_template()}
    stages = ann.get("stage_caisr")
    if stages is None or len(stages) == 0:
        return feat, None
    stages = stages.astype(int)
    valid = (stages >= 1) & (stages <= 5)
    vstages = stages[valid]
    n_epochs = len(vstages)
    if n_epochs == 0:
        return feat, None
    feat["arch_n_epochs"] = float(len(stages))
    for sname, sval in STAGES.items():
        feat[f"arch_frac_{sname}"] = float(np.mean(vstages == sval))
    sleep = np.isin(vstages, SLEEP_STAGE_VALUES)
    feat["arch_total_sleep_time_h"] = float(sleep.sum() * EPOCH_SECONDS / 3600.0)
    feat["arch_sleep_efficiency"] = float(sleep.mean()) if len(vstages) else np.nan
    # transitions (5x5, row-normalized)
    stage_map = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
    mapped = np.array([stage_map.get(s, -1) for s in vstages])
    trans = np.zeros((5, 5))
    for a, b in zip(mapped[:-1], mapped[1:]):
        if a >= 0 and b >= 0:
            trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums
    names = ["W", "N1", "N2", "N3", "R"]
    template_names = {"W": "wake", "N1": "n1", "N2": "n2", "N3": "n3", "R": "rem"}
    for i, sn in enumerate(names):
        for j, dn in enumerate(names):
            feat[f"trans_{template_names[sn]}_{template_names[dn]}"] = float(trans[i, j])

    # bouts + awakenings
    def _bout(mask_val):
        b = (vstages == mask_val).astype(int)
        d = np.diff(b, prepend=0, append=0)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        lens = ends - starts
        return (float(np.mean(lens)), float(np.std(lens))) if len(lens) else (np.nan, np.nan)

    mw, sw = _bout(5)
    mr, sr = _bout(4)
    feat["arch_bout_mean_W"], feat["arch_bout_std_W"] = mw, sw
    feat["arch_bout_mean_R"], feat["arch_bout_std_R"] = mr, sr
    in_sleep = np.flatnonzero(sleep)
    if len(in_sleep) > 2:
        period = vstages[in_sleep[0] : in_sleep[-1] + 1]
        feat["arch_n_awakenings"] = float(np.count_nonzero(np.diff((period == 5).astype(int), prepend=0) == 1))
    else:
        feat["arch_n_awakenings"] = 0.0
    feat["arch_n_rem_bouts"] = float(np.count_nonzero(np.diff((vstages == 4).astype(int), prepend=0) == 1))
    n3_idx = np.flatnonzero(vstages == 1)
    if n3_idx.size:
        half = len(vstages) // 2
        first = np.mean(n3_idx < half)
        feat["arch_n3_first_minus_second_frac"] = float(first - (1.0 - first))
    # entropies
    if n_epochs > 1:
        _, counts = np.unique(vstages, return_counts=True)
        feat["arch_stage_entropy"] = float(scipy_entropy(counts / counts.sum()))
    prob_keys = ["caisr_prob_n3", "caisr_prob_n2", "caisr_prob_n1", "caisr_prob_r", "caisr_prob_w"]
    if all(k in ann for k in prob_keys) and len(stages) == len(ann[prob_keys[0]]):
        probs = np.stack([ann[k] / 9.0 for k in prob_keys], axis=1)
        probs = np.clip(probs, 0, 1)
        sums = probs.sum(axis=1, keepdims=True)
        ok = (sums[:, 0] > 1e-6) & valid
        if ok.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = np.log(np.where(probs > 0, probs, 1.0))
            ent = -np.sum(probs * logp, axis=1)
            feat["arch_stage_prob_entropy_mean"] = float(np.mean(ent[ok]))
            feat["arch_stage_prob_entropy_std"] = float(np.std(ent[ok]))
    # event indices (events per hour)
    hours = float(len(stages) * EPOCH_SECONDS / 3600.0) or np.nan
    if hours and hours > 0:
        for key, name, val in [
            ("arousal_caisr", "arch_arousal_index", None),
            ("resp_caisr", "arch_ahi", None),
            ("limb_caisr", "arch_plmi", 2),
        ]:
            sig = ann.get(key)
            if sig is not None and len(sig):
                if val is None:
                    events = np.count_nonzero(np.diff((sig > 0).astype(int), prepend=0) == 1)
                else:
                    events = np.count_nonzero(np.diff((sig == val).astype(int), prepend=0) == 1)
                feat[name] = float(events / hours)
    return feat, stages


def _temporal_pooling(powers: np.ndarray, stages: np.ndarray) -> Dict[str, float]:
    feat = {k: np.nan for k in _temporal_template()}
    if powers is None or len(powers) == 0:
        return feat
    n = min(len(powers), len(stages))
    powers, stages = powers[:n], np.asarray(stages[:n])
    keep = np.isin(stages, SLEEP_STAGE_VALUES)
    p = powers[keep]
    if p.shape[0] < 20:
        return feat
    band_names = list(BANDS)
    for j, band in enumerate(band_names):
        v = p[:, j]
        v = v[np.isfinite(v)]
        if v.size < 20:
            continue
        for q in QUANTILES:
            feat[f"tp_{band}_q{int(round(q * 100))}"] = float(np.quantile(v, q))
        median = float(np.median(v))
        feat[f"tp_{band}_iqr"] = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
        feat[f"tp_{band}_range80"] = float(np.quantile(v, 0.90) - np.quantile(v, 0.10))
        feat[f"tp_{band}_cv"] = float(v.std() / median) if median > 1e-12 else np.nan
        third = max(1, len(v) // 3)
        feat[f"tp_{band}_drift"] = float(v[-third:].mean() - v[:third].mean())
        hi = v >= np.quantile(v, 0.88)
        window = max(2, int(3600 / EPOCH_SECONDS))
        if len(v) >= window:
            counts = np.convolve(hi.astype(float), np.ones(window), "valid")
            feat[f"tp_{band}_worst_hour_frac"] = float(counts.max() / window)
        else:
            feat[f"tp_{band}_worst_hour_frac"] = float(hi.mean())
        feat[f"tp_{band}_excursion_rate"] = float(np.mean(np.diff(hi.astype(int)) == 1) * (3600 / EPOCH_SECONDS))
    # high-quantile ratios (worst part of the night)
    if p.shape[1] == len(band_names):
        for num, den, ratio in [
            ("delta", "alpha", "delta_alpha"),
            ("theta", "alpha", "theta_alpha"),
            ("delta", "beta", "delta_beta"),
        ]:
            i, j = band_names.index(num), band_names.index(den)
            num_q = np.quantile(p[:, i][np.isfinite(p[:, i])], 0.88)
            den_q = np.quantile(p[:, j][np.isfinite(p[:, j])], 0.88)
            feat[f"tp_ratio_{ratio}_q88"] = float(num_q / den_q) if den_q > 1e-12 else np.nan
    return feat


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------


def process_record(args: Tuple[dict, str, str, dict]) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    row, raw_path, ann_path, rename_rules = args
    rec = str(row["BidsFolder"])
    feat = OrderedDict(FEATURE_TEMPLATE)
    meta = OrderedDict(META_TEMPLATE)

    meta["meta_age"] = row.get("Age")
    meta["meta_sex_male"] = 1.0 if str(row.get("Sex")).lower().startswith("male") else 0.0
    meta["meta_bmi"] = row.get("BMI")
    meta["meta_site"] = row.get("SiteID")
    meta["label"] = row.get("Cognitive_Impairment")
    try:
        meta["meta_rec_year"] = float(pd.to_datetime(row.get("CreationTime"), errors="coerce").year)
    except Exception:
        meta["meta_rec_year"] = np.nan

    ann = load_caisr(ann_path)
    arch, stages = _caisr_arch(ann) if ann is not None else ({k: np.nan for k in _arch_template()}, None)
    feat.update(arch)

    # raw channel plan
    try:
        with pyedflib.EdfReader(raw_path) as edf:
            raw_labels = edf.getSignalLabels()
    except Exception:
        raw_labels = []
    rename_map, _ = helper_code.standardize_channel_names_rename_only(list(raw_labels), rename_rules)
    std_to_orig = {}
    for orig, std in rename_map.items():
        std_to_orig.setdefault(std, orig)

    wanted_raw = set()
    for std in std_to_orig:
        if std in EEG_TARGETS or std in ("ecg", "spo2") or std in {m for p in MONO_PAIRS.values() for m in p}:
            wanted_raw.add(std_to_orig[std])
    raw = read_raw_channels(raw_path, wanted_raw)

    # build bipolar EEG signals {target: (sig, fs)}
    eeg: Dict[str, Tuple[np.ndarray, float]] = {}
    for target in EEG_TARGETS:
        if target in std_to_orig and std_to_orig[target] in raw:
            eeg[target] = raw[std_to_orig[target]]
        else:
            active, ref = MONO_PAIRS[target]
            a_raw = std_to_orig.get(active)
            r_raw = std_to_orig.get(ref)
            if a_raw and r_raw and a_raw in raw and r_raw in raw:
                sig_a, fs_a = raw[a_raw]
                sig_r, fs_r = raw[r_raw]
                if abs(fs_a - fs_r) < 1e-6:
                    n = min(len(sig_a), len(sig_r))
                    eeg[target] = (sig_a[:n] - sig_r[:n], fs_a)

    # per-epoch powers for the first two available EEG channels (temporal pooling)
    tp_channels = [c for c in EEG_TARGETS if c in eeg][:2]
    if stages is not None and tp_channels:
        sig, fs = eeg[tp_channels[0]]
        epoch_powers = _epoch_band_powers(sig, fs, stages)
        if epoch_powers is not None:
            feat.update(_temporal_pooling(epoch_powers, stages))

    # per-stage spectral + scalars
    if stages is not None:
        for ch in EEG_TARGETS:
            if ch not in eeg:
                continue
            sig, fs = eeg[ch]
            for sname, sval in STAGES.items():
                seg = _stage_concat(sig, fs, stages, sval, MAX_STAGE_CONCAT_SECONDS)
                if seg is None:
                    continue
                powers, edge95 = _band_power_relative(seg, fs)
                if powers is not None:
                    for band in BANDS:
                        feat[f"spec_{sname}_{band}_{ch}"] = powers[band]
                    if powers["alpha"] > 1e-12:
                        feat[f"spec_{sname}_theta_alpha_{ch}"] = powers["theta"] / powers["alpha"]
                        feat[f"spec_{sname}_delta_alpha_{ch}"] = powers["delta"] / powers["alpha"]
                    if powers["beta"] > 1e-12:
                        feat[f"spec_{sname}_delta_beta_{ch}"] = powers["delta"] / powers["beta"]
                    feat[f"spec_{sname}_edge95_{ch}"] = edge95
                feat[f"spec_{sname}_kurtosis_{ch}"] = _clipped_kurtosis(seg)
                act, mob, comp = _hjorth(seg)
                feat[f"spec_{sname}_hj_activity_{ch}"] = act
                feat[f"spec_{sname}_hj_mobility_{ch}"] = mob
                feat[f"spec_{sname}_hj_complexity_{ch}"] = comp

    # coherence
    if stages is not None:
        for a, b in COHERENCE_PAIRS:
            if a not in eeg or b not in eeg:
                continue
            sig_a, fs_a = eeg[a]
            sig_b, fs_b = eeg[b]
            if abs(fs_a - fs_b) >= 1e-6:
                continue
            for sname, sval in STAGES.items():
                seg_a = _stage_concat(sig_a, fs_a, stages, sval, MAX_COHERENCE_SECONDS)
                seg_b = _stage_concat(sig_b, fs_b, stages, sval, MAX_COHERENCE_SECONDS)
                if seg_a is None or seg_b is None:
                    continue
                coh = _pair_coherence(seg_a, seg_b, fs_a)
                if coh is None:
                    continue
                for band, val in coh.items():
                    feat[f"coh_{a.replace('-', '')}_{b.replace('-', '')}_{band}_{sname}"] = val

    # HRV
    if stages is not None and "ecg" in std_to_orig and std_to_orig["ecg"] in raw:
        ecg_sig, ecg_fs = raw[std_to_orig["ecg"]]
        epoch_len = int(round(EPOCH_SECONDS * ecg_fs)) if ecg_fs > 0 else 0
        masks = {
            "nrem": np.isin(np.asarray(stages[: len(stages)]), (1, 2, 3)),
            "rem": np.asarray(stages[: len(stages)]) == 4,
        }
        feat.update(_hrv_features(ecg_sig, ecg_fs, masks, epoch_len))

    # SpO2
    if stages is not None and "spo2" in std_to_orig and std_to_orig["spo2"] in raw:
        spo2_sig, spo2_fs = raw[std_to_orig["spo2"]]
        feat.update(_spo2_features(spo2_sig, spo2_fs, len(stages)))

    return rec, feat, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_tasks(
    data_root: Path,
    demo: pd.DataFrame,
    limit: Optional[int],
    require_raw: bool = True,
) -> List[Tuple[dict, str, str, dict]]:
    rules = helper_code.load_rename_rules(str(Path(__file__).resolve().parent.parent / "channel_table.csv"))
    tasks = []
    for _, row in demo.iterrows():
        base = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        site = row["SiteID"]
        raw_path = data_root / "physiological_data" / site / f"{base}.edf"
        ann_path = data_root / "algorithmic_annotations" / site / f"{base}_caisr_annotations.edf"
        if require_raw and not raw_path.exists():
            continue
        if not ann_path.exists():
            continue
        tasks.append((dict(row), str(raw_path) if raw_path.exists() else "", str(ann_path), rules))
        if limit and len(tasks) >= limit:
            break
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="limit records (smoke run)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--require-raw",
        dest="require_raw",
        action="store_true",
        default=True,
        help="skip records without raw PSG (default); use --no-require-raw for CAISR-only runs",
    )
    parser.add_argument("--no-require-raw", dest="require_raw", action="store_false")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = pd.read_csv(data_root / "demographics.csv")
    tasks = build_tasks(data_root, demo, args.limit, require_raw=args.require_raw)
    print(f"records to process: {len(tasks)}", flush=True)

    rows_feat, rows_meta = [], []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (rec, feat, meta) in enumerate(pool.map(process_record, tasks)):
                rows_feat.append(feat)
                rows_meta.append(meta)
                if (i + 1) % 25 == 0:
                    print(f"processed {i + 1}/{len(tasks)}", flush=True)
    else:
        for i, task in enumerate(tasks):
            rec, feat, meta = process_record(task)
            rows_feat.append(feat)
            rows_meta.append(meta)
            if (i + 1) % 10 == 0:
                print(f"processed {i + 1}/{len(tasks)}", flush=True)

    recs = [str(t[0]["BidsFolder"]) for t in tasks]
    feat_df = pd.DataFrame(rows_feat, index=recs)
    meta_df = pd.DataFrame(rows_meta, index=recs)
    feat_df.to_csv(out_dir / "features.csv")
    meta_df.to_csv(out_dir / "record_meta.csv")

    # NaN report per feature, overall + per site
    site = meta_df["meta_site"]
    nan_all = feat_df.isna().mean()
    report = pd.DataFrame({"nan_rate_overall": nan_all})
    for s in sorted(site.dropna().unique()):
        mask = (site == s).values
        report[f"nan_rate_{s}"] = feat_df[mask].isna().mean()
    report.to_csv(out_dir / "nan_report.csv")
    n_complete = int((feat_df.isna().sum(axis=1) == 0).sum())
    print(f"done: {len(recs)} records, {feat_df.shape[1]} features, {n_complete} fully complete", flush=True)
    print(f"outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
