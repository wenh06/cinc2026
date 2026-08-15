"""Philosopher's Stone latent cache loading + on-the-fly fallback.

The BDSP brain-health model (``third_party/philosophers-stone``) costs ~2-4
minutes per recording, so the submission reads precomputed 1024-D latents from
a cache directory and only runs the model on cache misses (e.g. when the
organisers perturb the training set during robustness checks).

Cache layout (produced by ``scripts/phi_cache_extract.py``)::

    <cache_dir>/<SiteID>/<BidsFolder>__<SessionID>.npz

with keys ``latent`` (1024, float32), ``brain_health_score``,
``total_cognition_score``, ``fluid_cognition_score``,
``crystallized_cognition_score`` and optionally ``heads``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyedflib

PHI_LATENT_DIM = 1024
PHI_SCORE_KEYS = (
    "brain_health_score",
    "total_cognition_score",
    "fluid_cognition_score",
    "crystallized_cognition_score",
)


def record_key(row: pd.Series) -> str:
    """Stable per-record key used by the cache files."""
    return f"{row['BidsFolder']}__{row['SessionID']}"


def cache_path_for_row(cache_dir: str | Path, row: pd.Series) -> Path:
    return Path(cache_dir) / str(row["SiteID"]) / f"{record_key(row)}.npz"


def load_phi_cache(cache_dir: str | Path, demo: pd.DataFrame) -> pd.DataFrame:
    """Load cached Phi latents/scores for every row of ``demo``.

    Returns a DataFrame indexed like ``demo`` with columns ``lhl_0..lhl_1023``
    plus the four scores; missing records are NaN.
    """
    cache_dir = Path(cache_dir)
    latents = np.full((len(demo), PHI_LATENT_DIM), np.nan, dtype=np.float32)
    scores = np.full((len(demo), len(PHI_SCORE_KEYS)), np.nan, dtype=np.float32)
    for i, (_, row) in enumerate(demo.iterrows()):
        path = cache_path_for_row(cache_dir, row)
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as npz:
                latents[i] = npz["latent"].astype(np.float32)
                for j, key in enumerate(PHI_SCORE_KEYS):
                    scores[i, j] = float(npz[key])
        except Exception:  # corrupt cache file — treat as miss
            continue
    out = pd.DataFrame(
        latents,
        index=demo.index,
        columns=[f"lhl_{k}" for k in range(PHI_LATENT_DIM)],
    )
    for j, key in enumerate(PHI_SCORE_KEYS):
        out[f"phi_{key}"] = scores[:, j]
    return out


def _resolve_c4m1(edf_path: str | Path) -> Optional[Tuple[np.ndarray, float]]:
    """Return (C4-M1 in µV, fs) for a raw EDF.

    Direct bipolar C4-M1/C4-A1 first; otherwise derive C4 − M1 from monopolar
    channels (I0006); last resort a bare C4.
    """
    aliases = ("c4-m1", "c4m1", "c4-a1", "c4a1")
    try:
        with pyedflib.EdfReader(str(edf_path)) as edf:
            labels = [lab.lower().strip() for lab in edf.getSignalLabels()]

            def norm(s: str) -> str:
                return s.lower().replace(" ", "").replace("-", "").replace("+", "")

            norm_idx = {norm(lab): i for i, lab in enumerate(labels)}
            for alias in aliases:
                if norm(alias) in norm_idx:
                    i = norm_idx[norm(alias)]
                    return edf.readSignal(i).astype(np.float64), float(edf.getSampleFrequency(i))
            # monopolar derivation
            if "c4" in norm_idx:
                c4 = edf.readSignal(norm_idx["c4"]).astype(np.float64)
                fs = float(edf.getSampleFrequency(norm_idx["c4"]))
                if "m1" in norm_idx:
                    m1 = edf.readSignal(norm_idx["m1"]).astype(np.float64)
                    n = min(len(c4), len(m1))
                    return c4[:n] - m1[:n], fs
                return c4, fs
    except Exception:
        return None
    return None


def compute_phi_on_the_fly(
    edf_path: str | Path,
    age: float,
    sex_male: int,
    model_file: str | Path,
    device_id: int = 0,
    collect_heads: bool = True,
) -> Dict[str, object]:
    """Compute Phi latent/scores for one raw EDF via the array API.

    Used only as a cache-miss fallback (slow: minutes per recording).
    """
    eeg = _resolve_c4m1(edf_path)
    if eeg is None:
        raise RuntimeError(f"no usable C4-M1 EEG in {edf_path}")
    signal, fs = eeg
    if not np.isfinite(signal).all():
        raise RuntimeError(f"non-finite EEG in {edf_path}")

    phi_root = Path(__file__).resolve().parent.parent / "third_party" / "philosophers-stone"
    src = phi_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from philosophers_stone.philosopher_utils import Config, infer_brain_health, load_model

    cfg = Config(model_file=str(model_file))
    cfg.device = f"cuda:{device_id}" if __import__("torch").cuda.is_available() else "cpu"
    model = load_model(cfg)
    result = infer_brain_health(
        signal,
        fs_hz=fs,
        age=float(age),
        sex=int(sex_male),
        file_id=os.path.basename(str(edf_path)),
        cfg=cfg,
        model=model,
        collect_head_outputs=collect_heads,
    )
    if result.get("status") != "ok":
        raise RuntimeError(str(result.get("error_message", "Phi inference failed")))
    return result
