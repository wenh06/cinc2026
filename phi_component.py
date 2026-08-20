#!/usr/bin/env python
"""Phi component: frozen Philosopher's Stone latents -> PCA -> tabular ranker.

This is the sub6/sub7 primary model.  Latents are cache-first: records present in
the mounted data folder read their 1024-dim latent from ``TrainCfg.phi.cache``
(vendored at ``data/phi_cache`` in the image); cache misses are computed on the
fly from the raw EDF using the baked checkpoint.  A record without a usable
C4-M1 (or a non-finite EEG) raises at inference time, so the component chain
falls through to the next component — the montage-agnostic sub5 tabular XGB.

The ranker is trained from scratch on the mounted records' latents (PCA fit +
booster fit), so the official train -> inference -> eval round-trip genuinely
depends on the provided data; cached latents for records absent from the
mounted data folder are never used.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from component_registry import ARTIFACT_NAMES
from helper_code import DEMOGRAPHICS_FILE, HEADERS, load_demographics, load_rename_rules
from tabular_pipeline import (
    _import_process_record,
    _load_feature_cache,
    _record_key,
    assemble_record_features,
)
from utils.phi_cache import (
    PHI_LATENT_DIM,
    PHI_SCORE_KEYS,
    cache_path_for_row,
    compute_phi_on_the_fly,
)

PHI_CKPT_NAME = "SleepPhilosophersStone.ckpt"


def _xgb_n_jobs() -> int:
    """Thread count for the XGBoost ranker (env-overridable; default all cores)."""
    return int(os.environ.get("CINC2026_XGB_N_JOBS", "-1"))


def resolve_phi_cache(train_config: Any) -> None:
    """Fill ``phi.cache`` from the vendored repo cache when unset.

    Mirrors :func:`component_registry.resolve_feature_cache`: an explicit path
    wins; otherwise the image-baked ``<repo>/data/phi_cache`` is used.  Only
    reads files bundled into the image — no network, no writes outside
    ``model_folder``.
    """
    phi = train_config.get("phi", None)
    if phi is None:
        return
    if str(phi.get("cache", "") or "").strip():
        return
    candidate = Path(__file__).resolve().parent / "data" / "phi_cache"
    if candidate.is_dir():
        phi.cache = str(candidate)


def resolve_phi_checkpoint(train_config: Any) -> Path:
    """Return the Philosopher's Stone checkpoint path (explicit config wins)."""
    phi = train_config.get("phi", {})
    explicit = str(phi.get("checkpoint", "") or "").strip()
    if explicit:
        return Path(explicit)
    cache_root = Path(os.environ.get("MODEL_CACHE_DIR", "/challenge/cache/revenger_model_dir"))
    return cache_root / "philosophers-stone" / "model_files" / PHI_CKPT_NAME


def _load_latent_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (latent float32 (1024,), scores float32 (4,)) from a cache npz."""
    with np.load(path, allow_pickle=False) as npz:
        latent = np.asarray(npz["latent"], dtype=np.float32).reshape(-1)
        if latent.shape[0] != PHI_LATENT_DIM:
            raise ValueError(f"bad latent dim {latent.shape[0]} (expected {PHI_LATENT_DIM})")
        scores = np.asarray([float(npz[k]) for k in PHI_SCORE_KEYS], dtype=np.float32)
    return latent, scores


def _phi_patient_data(demo_file: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    """Demographics for one record; SessionID must stay raw (int64 in the CSV).

    ``load_demographics`` compares the SessionID column strictly, so a str
    never matches the int64 values and silently yields an empty dict, which
    propagates a NaN age into the brain-health model and produces an all-NaN
    latent on cache-miss records (the official hidden set).  Only the raw
    SessionID is valid here.
    """
    return load_demographics(
        str(demo_file),
        record[HEADERS["bids_folder"]],
        record[HEADERS["session_id"]],
    )


def _spec_sources(train_config: Any) -> Dict[str, Any]:
    """Spectral-block sources shared with the tabular path (cache-first)."""
    tab = train_config.get("tabular") or {}
    frame, lookup = _load_feature_cache(str(tab.get("feature_cache", "") or ""))
    return {
        "frame": frame,
        "lookup": lookup,
        "rename_rules": load_rename_rules(str(PROJECT_ROOT / "channel_table.csv")),
        "feature_groups": list(tab.get("feature_groups") or ["spec"]),
    }


def _spectral_row(
    record: Dict[str, str],
    patient_data: Dict[str, Any],
    data_folder: str,
    sources: Dict[str, Any],
) -> pd.Series:
    """One record's full spectral feature row (cache-first; on-the-fly fallback)."""
    bids = str(record[HEADERS["bids_folder"]])
    session = str(record[HEADERS["session_id"]])
    site = str(record[HEADERS["site_id"]])
    idx = sources["lookup"].get(_record_key(bids, session))
    if idx is None:
        idx = sources["lookup"].get(bids)
    if idx is not None and sources["frame"] is not None:
        return sources["frame"].iloc[idx]
    base = f"{bids}_ses-{session}"
    raw_path = Path(data_folder) / "physiological_data" / site / f"{base}.edf"
    ann_path = Path(data_folder) / "algorithmic_annotations" / site / f"{base}_caisr_annotations.edf"
    if not raw_path.is_file():
        raise ValueError(f"fusion: no raw EDF for {bids} and spectral cache miss")
    process_record = _import_process_record()
    _rec, feat_dict, _meta = process_record((dict(patient_data), str(raw_path), str(ann_path), sources["rename_rules"]))
    return pd.Series(feat_dict)


def _compute_record_on_the_fly(
    record: Dict[str, str],
    data_folder: str,
    checkpoint: Path,
    cache_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Cache-first latent fetch for one record; on-the-fly for misses."""
    bids = str(record[HEADERS["bids_folder"]])
    site = str(record[HEADERS["site_id"]])
    session = str(record[HEADERS["session_id"]])
    if cache_dir is not None:
        cache_path = cache_path_for_row(str(cache_dir), pd.Series(record))
        if cache_path.is_file():
            return _load_latent_npz(cache_path)

    raw_path = Path(data_folder) / "physiological_data" / site / f"{bids}_ses-{session}.edf"
    if not raw_path.is_file():
        raise ValueError(f"phi: no raw EDF for {bids} and cache miss")
    demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
    patient_data = _phi_patient_data(demo_file, record)
    try:
        age = float(patient_data.get(HEADERS["age"], np.nan))
    except (TypeError, ValueError):
        raise ValueError(f"phi: missing/invalid age for {bids}") from None
    sex_male = 1 if str(patient_data.get(HEADERS["sex"], "")).lower().startswith("male") else 0
    result = compute_phi_on_the_fly(
        raw_path,
        age=age,
        sex_male=sex_male,
        model_file=checkpoint,
        device_id=0,
        collect_heads=False,
    )
    latent = np.asarray(result["latent"], dtype=np.float32).reshape(-1)
    scores = np.asarray([float(result[k]) for k in PHI_SCORE_KEYS], dtype=np.float32)
    return latent, scores


def train_phi_model(train_config: Any, model_folder: Path, verbose: bool) -> None:
    """Train the PCA + ranker on the mounted records' latents; save artifacts."""
    phi = {k: v for k, v in train_config.get("phi").items()}
    data_root = Path(train_config.db_dir)
    demo = pd.read_csv(data_root / DEMOGRAPHICS_FILE)
    cache_dir = Path(phi.get("cache")) if str(phi.get("cache", "") or "").strip() else None
    checkpoint = resolve_phi_checkpoint(train_config)
    fusion = str(phi.get("features", "latent")).lower() == "fusion"
    spec_sources = _spec_sources(train_config) if fusion else None
    spec_series: list = []

    latent_list: list = []
    score_list: list = []
    labels: list = []
    n_cache = n_miss = 0
    for _, row in demo.iterrows():
        record = dict(row)
        try:
            latent, scores = _compute_record_on_the_fly(record, str(data_root), checkpoint, cache_dir)
            if not np.isfinite(latent).all():
                raise ValueError("non-finite latent")
            spec = _spectral_row(record, record, str(data_root), spec_sources) if fusion else None
            latent_list.append(latent)
            score_list.append(scores)
            if spec is not None:
                spec_series.append(spec)
            labels.append(int(bool(row.get(HEADERS["label"], False))))
            if cache_dir is not None and cache_path_for_row(str(cache_dir), row).is_file():
                n_cache += 1
            else:
                n_miss += 1
        except Exception as exc:  # noqa: BLE001 — record served by the fallback component
            if verbose:
                print(f"[CinC2026] phi: skipping {row['BidsFolder']} (no latent): {exc!r}")

    if not latent_list:
        raise RuntimeError("phi: no usable latents for any training record")
    if fusion and not spec_series:
        raise RuntimeError(
            "phi: fusion mode but no spectral features for any training record "
            "(cache empty and no raw EDFs) — the vendored spectral cache is shadowed?"
        )
    if verbose:
        print(f"[CinC2026] phi: {len(latent_list)} training rows (cache {n_cache}, on-the-fly {n_miss})")

    include_scores = bool(phi.get("include_scores", False))
    pca_dim = int(phi.get("pca_dim", 64))
    model_name = str(phi.get("model", "xgboost")).lower()
    threshold = float(phi.get("threshold", 0.5))
    X = np.stack(latent_list)
    S = np.stack(score_list)
    y = np.asarray(labels, dtype=int)

    config: Dict[str, Any] = {
        "model": model_name,
        "pca_dim": pca_dim,
        "include_scores": include_scores,
        "features": "fusion" if fusion else "latent",
        "threshold": threshold,
        "constant": None,
        "n_train": int(len(y)),
        "n_latent": int(X.shape[1]),
    }

    if len(set(y.tolist())) == 1:
        config["constant"] = float(y.mean())
        if verbose:
            print(f"[CinC2026] phi: single-class labels — constant model prob={config['constant']:.4f}")
    else:
        pca_dim = min(pca_dim, X.shape[1], int(X.shape[0]))
        config["pca_dim"] = pca_dim
        pca = PCA(n_components=pca_dim, random_state=0)
        Z = pca.fit_transform(X).astype(np.float32)
        if fusion:
            spec_frame = pd.DataFrame(spec_series)
            spec_list = [c for c in spec_frame.columns if not c.startswith("meta_")]
            groups = spec_sources["feature_groups"]
            if groups:
                spec_list = [c for c in spec_list if c.split("_")[0] in groups]
            Z = np.hstack([Z, spec_frame[spec_list].to_numpy(dtype=np.float32)]).astype(np.float32)
            config["spec_feature_list"] = spec_list
        if include_scores:
            Z = np.hstack([Z, S]).astype(np.float32)
        config["n_features"] = int(Z.shape[1])
        with open(model_folder / ARTIFACT_NAMES["phi"]["pca"], "wb") as fh:
            pickle.dump(pca, fh, protocol=4)

        if model_name in ("xgboost", "ensemble"):
            import xgboost as xgb

            params = dict(phi.get("xgb_params") or {})
            seed = int(params.pop("seed", params.pop("random_state", 0)))
            params.setdefault("tree_method", "hist")
            clf = xgb.XGBClassifier(random_state=seed, n_jobs=_xgb_n_jobs(), **params)
            clf.fit(Z, y)
            clf.get_booster().save_model(str(model_folder / ARTIFACT_NAMES["phi"]["xgboost"]))
        if model_name in ("logistic", "ensemble"):
            steps = ([SimpleImputer(strategy="median")] if fusion else []) + [
                StandardScaler(),
                LogisticRegression(**dict(phi.get("lr_params") or {})),
            ]
            pipeline = make_pipeline(*steps)
            pipeline.fit(Z, y)
            with open(model_folder / ARTIFACT_NAMES["phi"]["logistic"], "wb") as fh:
                pickle.dump(pipeline, fh, protocol=4)
        if model_name not in ("xgboost", "logistic", "ensemble"):
            raise ValueError(f"phi: unknown model {model_name!r}")

    (model_folder / ARTIFACT_NAMES["phi"]["config"]).write_text(json.dumps(config, indent=2))
    if verbose:
        print(f"[CinC2026] phi model saved to {model_folder}")


def load_phi_model(model_folder: Path, train_config: Any, verbose: bool) -> Dict[str, Any]:
    """Load the PCA + ranker artifacts into a component payload."""
    config_path = model_folder / ARTIFACT_NAMES["phi"]["config"]
    if not config_path.is_file():
        raise FileNotFoundError(f"phi config missing at {config_path}")
    config = json.loads(config_path.read_text())
    phi_cfg = train_config.get("phi", {})
    cache_str = str(phi_cfg.get("cache", "") or "").strip()
    payload: Dict[str, Any] = {
        "config": config,
        "cache_dir": Path(cache_str) if cache_str else None,
        "checkpoint": resolve_phi_checkpoint(train_config),
        "pca": None,
        "ranker": None,
    }
    if config.get("features") == "fusion":
        payload["spec_sources"] = _spec_sources(train_config)
    if config.get("constant") is None:
        with open(model_folder / ARTIFACT_NAMES["phi"]["pca"], "rb") as fh:
            payload["pca"] = pickle.load(fh)
        model_name = config.get("model")
        if model_name in ("xgboost", "ensemble"):
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(model_folder / ARTIFACT_NAMES["phi"]["xgboost"]))
            payload["ranker"] = {"xgb": booster} if model_name == "ensemble" else booster
        if model_name in ("logistic", "ensemble"):
            with open(model_folder / ARTIFACT_NAMES["phi"]["logistic"], "rb") as fh:
                lr_pipeline = pickle.load(fh)
            if model_name == "ensemble":
                payload["ranker"]["lr"] = lr_pipeline
            else:
                payload["ranker"] = lr_pipeline
    if verbose:
        print(
            f"[CinC2026] phi model loaded ({config.get('model')}, "
            f"n_train={config.get('n_train')}, pca_dim={config.get('pca_dim')})"
        )
    return payload


def run_phi_model(payload: Dict[str, Any], record: Dict[str, str], data_folder: str, verbose: bool) -> Tuple[int, float]:
    """Inference for one record through the PCA + ranker (cache-first)."""
    config = payload["config"]
    threshold = float(config.get("threshold", 0.5))
    if config.get("constant") is not None:
        p = float(config["constant"])
        return int(p >= threshold), p

    latent, scores = _compute_record_on_the_fly(
        record,
        data_folder,
        Path(payload["checkpoint"]),
        payload.get("cache_dir"),
    )
    if not np.isfinite(latent).all():
        # Never feed NaN into the ranker: XGBoost turns an all-NaN row into a
        # constant probability, which silently collapses age-cond to ~0.5.
        # Raising routes the record to the next component in the chain.
        raise ValueError(f"phi: non-finite latent for {record[HEADERS['bids_folder']]}")
    z = payload["pca"].transform(latent.reshape(1, -1)).astype(np.float32)
    if config.get("features") == "fusion":
        demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
        patient_data = _phi_patient_data(demo_file, record)
        feat = _spectral_row(record, patient_data, data_folder, payload["spec_sources"])
        spec_row = assemble_record_features(
            feat,
            patient_data,
            {"feature_list": config["spec_feature_list"], "include_meta": False},
        )
        z = np.hstack([z, spec_row.to_numpy(dtype=np.float32).reshape(1, -1)]).astype(np.float32)
    if config.get("include_scores"):
        z = np.hstack([z, scores.reshape(1, -1)]).astype(np.float32)

    if config.get("model") == "ensemble":
        import xgboost as xgb

        p_lr = float(payload["ranker"]["lr"].predict_proba(z)[0, 1])
        margin = payload["ranker"]["xgb"].predict(xgb.DMatrix(z))
        p_xgb = float(1.0 / (1.0 + np.exp(-margin[0])))
        p = 0.5 * (p_lr + p_xgb)
    elif config.get("model") == "xgboost":
        import xgboost as xgb

        margin = payload["ranker"].predict(xgb.DMatrix(z))
        p = float(1.0 / (1.0 + np.exp(-margin[0])))
    else:
        p = float(payload["ranker"].predict_proba(z)[0, 1])
    return int(p >= threshold), p
