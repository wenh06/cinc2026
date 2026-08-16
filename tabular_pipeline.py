#!/usr/bin/env python
"""
Tabular (D2) submission path — 641-dim spectral/physiological feature bank
plus gradient-boosted trees (XGBoost / LightGBM).

Kept out of ``team_code.py`` so the default CRNN entry stays untouched; the
entry module imports the four functions below and routes to them when
``TrainCfg.tabular.enable`` is True:

    tabular_enabled(train_config) -> bool
    train_tabular(train_config, model_folder, verbose)
    load_tabular_model(model_folder, train_config, verbose) -> model_dict
    run_tabular_model(model_dict, record, data_folder, verbose) -> (binary, prob)

Features are cache-first: a D1-layout ``features.csv`` (index = BidsFolder or
``BidsFolder__SessionID``) may be supplied via ``TrainCfg.tabular.feature_cache``;
cache misses are extracted on the fly from the raw PSG + CAISR annotations
(multiprocess).  The fitted feature-column list is serialised with the model,
so training and inference always see identical columns.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from helper_code import (
    DEMOGRAPHICS_FILE,
    HEADERS,
    load_demographics,
    load_rename_rules,
)

_TABULAR_CONFIG_NAME = "tabular_config.json"
_TABULAR_XGB_MODEL_NAME = "tabular_model.json"
_TABULAR_LGB_MODEL_NAME = "tabular_model.txt"

_EXTRACTOR_PROCESS_RECORD = None


def _xgb_n_jobs() -> int:
    """XGBoost thread count; env-overridable for shared/oversubscribed hosts.

    ``CINC2026_XGB_N_JOBS=-1`` (default) lets XGBoost use all cores — fine on a
    dedicated machine (the official scorer).  Set it to a small integer when
    training on a shared box, where ``n_jobs=-1`` thrashes badly (measured
    >70 s vs 0.34 s for the same fit at ``n_jobs=2``).
    """
    raw = os.environ.get("CINC2026_XGB_N_JOBS", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return -1


def tabular_enabled(train_config: Any) -> bool:
    """Return True when the tabular submission path is switched on."""
    tab = train_config.get("tabular", None)
    return tab is not None and bool(tab.get("enable", False))


def _record_key(bids_folder: str, session_id: str) -> str:
    return f"{bids_folder}__{session_id}"


def _meta_vector(patient_data: Dict) -> Dict[str, float]:
    """age / sex / bmi / recording-year — the four D2 metadata columns."""
    import pandas as pd

    try:
        age = float(patient_data.get(HEADERS["age"]))
    except (TypeError, ValueError):
        age = np.nan
    sex = 1.0 if str(patient_data.get(HEADERS["sex"], "")).strip().lower().startswith("m") else 0.0
    try:
        bmi = float(patient_data.get(HEADERS["bmi"]))
    except (TypeError, ValueError):
        bmi = np.nan
    try:
        rec_year = float(pd.to_datetime(patient_data.get(HEADERS["creation_time"]), errors="coerce").year)
    except Exception:
        rec_year = np.nan
    return {"meta_age": age, "meta_sex_male": sex, "meta_bmi": bmi, "meta_rec_year": rec_year}


def assemble_record_features(feat, patient_data: Dict, config: Dict[str, Any]):
    """Assemble one record's feature row in the trained column order.

    Fills the four metadata columns from demographics when
    ``config["include_meta"]`` is set (identical to the training-time
    construction) and reindexes to ``config["feature_list"]``.  Exposed so the
    inference contract is unit-testable — ``test_docker.test_tabular`` guards
    against metadata being silently left NaN.
    """
    import pandas as pd

    if not isinstance(feat, pd.Series):
        feat = pd.Series(feat)
    if config.get("include_meta"):
        for col, val in _meta_vector(patient_data).items():
            feat[col] = val
    return feat.reindex(config["feature_list"])


def _load_feature_cache(cache_path: str) -> Tuple[Optional[Any], Dict[str, int]]:
    """Load a D1-layout features.csv (index = BidsFolder or BidsFolder__SessionID)."""
    import pandas as pd

    if not cache_path:
        return None, {}
    path = Path(cache_path)
    if not path.exists():
        return None, {}
    frame = pd.read_csv(path, index_col=0)
    lookup: Dict[str, int] = {}
    for i, idx in enumerate(frame.index):
        key = str(idx)
        lookup[key] = i
        lookup.setdefault(key.split("__")[0], i)
    return frame, lookup


def _import_process_record():
    """Lazy-import the D1 extractor's per-record worker (heavy scipy imports)."""
    global _EXTRACTOR_PROCESS_RECORD
    if _EXTRACTOR_PROCESS_RECORD is None:
        import sys

        scripts_dir = str(Path(__file__).resolve().parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from extract_spectral_features import process_record

        _EXTRACTOR_PROCESS_RECORD = process_record
    return _EXTRACTOR_PROCESS_RECORD


def _compute_many(data_root: Path, rows: List[Dict[str, Any]], workers: int) -> Dict[str, Dict[str, float]]:
    """Extract 641-dim features on the fly for cache misses (raw EDF required)."""
    process_record = _import_process_record()
    rules = load_rename_rules(str(Path(__file__).resolve().parent / "channel_table.csv"))
    tasks = []
    for row in rows:
        base = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        raw_path = data_root / "physiological_data" / str(row["SiteID"]) / f"{base}.edf"
        ann_path = data_root / "algorithmic_annotations" / str(row["SiteID"]) / f"{base}_caisr_annotations.edf"
        if raw_path.exists():
            tasks.append((dict(row), str(raw_path), str(ann_path), rules))
    out: Dict[str, Dict[str, float]] = {}
    if not tasks:
        return out
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for task, (_rec, feat, _meta) in zip(tasks, pool.map(process_record, tasks)):
                key = _record_key(str(task[0]["BidsFolder"]), str(task[0]["SessionID"]))
                out[key] = dict(feat)
    else:
        for task in tasks:
            _rec, feat, _meta = process_record(task)
            key = _record_key(str(task[0]["BidsFolder"]), str(task[0]["SessionID"]))
            out[key] = dict(feat)
    return out


def train_tabular(train_config: Any, model_folder: Path, verbose: bool) -> None:
    """Train the tabular path: spectral-bank features → XGBoost/LightGBM classifier."""
    import pandas as pd

    tab = {k: v for k, v in train_config.get("tabular").items()}
    data_root = Path(train_config.db_dir)
    demo = pd.read_csv(data_root / DEMOGRAPHICS_FILE)
    cache_frame, cache_lookup = _load_feature_cache(tab.get("feature_cache", ""))

    feature_groups = list(tab.get("feature_groups") or [])
    include_meta = bool(tab.get("include_meta", True))

    rows: List[Dict[str, Any]] = []
    feat_series = []
    labels: List[int] = []
    missing_rows: List[Dict[str, Any]] = []
    for _, raw_row in demo.iterrows():
        row = dict(raw_row)
        bids, sess = str(row["BidsFolder"]), str(row["SessionID"])
        idx = cache_lookup.get(_record_key(bids, sess))
        if idx is None:
            idx = cache_lookup.get(bids)
        if idx is not None and cache_frame is not None:
            rows.append(row)
            feat_series.append(cache_frame.iloc[idx])
            labels.append(int(bool(row.get("Cognitive_Impairment", False))))
        else:
            missing_rows.append(row)

    on_the_fly = _compute_many(data_root, missing_rows, int(tab.get("workers", 4)))
    for row in missing_rows:
        key = _record_key(str(row["BidsFolder"]), str(row["SessionID"]))
        if key in on_the_fly:
            rows.append(row)
            feat_series.append(pd.Series(on_the_fly[key]))
            labels.append(int(bool(row.get("Cognitive_Impairment", False))))

    if not rows:
        raise RuntimeError("tabular: no records with features (cache empty and no raw EDFs)")
    if verbose:
        n_cache = len(feat_series) - len(on_the_fly)
        print(f"[CinC2026] tabular: {len(rows)} feature rows (cache {n_cache}, on-the-fly {len(on_the_fly)})")

    x = pd.DataFrame(feat_series)
    feature_list = [c for c in x.columns if not c.startswith("meta_")]
    if feature_groups:
        feature_list = [c for c in feature_list if c.split("_")[0] in feature_groups]
    x = x[feature_list].reset_index(drop=True)
    meta_cols = ["meta_age", "meta_sex_male", "meta_bmi", "meta_rec_year"]
    if include_meta:
        meta_df = pd.DataFrame([_meta_vector(r) for r in rows], columns=meta_cols)
        x = pd.concat([x, meta_df], axis=1)
        feature_list = feature_list + meta_cols
    y = np.asarray(labels, dtype=int)

    model_name = str(tab.get("model", "xgboost")).lower()
    threshold = float(tab.get("threshold", 0.5))
    constant = None
    if model_name == "xgboost":
        import xgboost as xgb

        params = dict(tab.get("xgb_params") or {})
        random_state = int(params.pop("seed", params.pop("random_state", 0)))
        params.setdefault("tree_method", "hist")
        clf = xgb.XGBClassifier(random_state=random_state, n_jobs=_xgb_n_jobs(), **params)
    elif model_name == "lightgbm":
        import lightgbm as lgb

        params = dict(tab.get("lgbm_params") or {})
        random_state = int(params.pop("seed", params.pop("random_state", 0)))
        clf = lgb.LGBMClassifier(random_state=random_state, **params)
    else:
        raise ValueError(f"tabular: unknown model {model_name!r}")

    if len(set(y.tolist())) < 2:
        constant = float(y.mean())
        if verbose:
            print(f"[CinC2026] tabular: single-class labels — constant model prob={constant:.4f}")
    else:
        clf.fit(x, y)

    config = {
        "model": model_name,
        "feature_list": feature_list,
        "threshold": threshold,
        "include_meta": include_meta,
        "constant": constant,
        "n_train": int(len(rows)),
    }
    model_folder.mkdir(parents=True, exist_ok=True)
    with open(model_folder / _TABULAR_CONFIG_NAME, "w") as f:
        json.dump(config, f, indent=2)
    if constant is None:
        if model_name == "xgboost":
            clf.get_booster().save_model(str(model_folder / _TABULAR_XGB_MODEL_NAME))
        else:
            clf.booster_.save_model(str(model_folder / _TABULAR_LGB_MODEL_NAME))
    if verbose:
        print(f"[CinC2026] tabular model saved to {model_folder}")


def load_tabular_model(model_folder: Path, train_config: Any, verbose: bool) -> Dict[str, Any]:
    """Load the tabular artefacts into a model_dict consumed by run_model."""
    cfg_path = model_folder / _TABULAR_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"tabular config missing at {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    booster = None
    if config.get("constant") is None:
        if config.get("model") == "xgboost":
            import xgboost as xgb

            booster = xgb.Booster(model_file=str(model_folder / _TABULAR_XGB_MODEL_NAME))
        else:
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(model_folder / _TABULAR_LGB_MODEL_NAME))
    tab = {k: v for k, v in train_config.get("tabular").items()}
    cache_frame, cache_lookup = _load_feature_cache(tab.get("feature_cache", ""))
    rename_rules = load_rename_rules(str(Path(__file__).resolve().parent / "channel_table.csv"))
    if verbose:
        print(f"[CinC2026] tabular model loaded ({config.get('model')}, n_train={config.get('n_train')})")
    return {
        "tabular": {
            "config": config,
            "booster": booster,
            "cache_frame": cache_frame,
            "cache_lookup": cache_lookup,
            "rename_rules": rename_rules,
        }
    }


def run_tabular_model(model_dict: Dict[str, Any], record: Dict[str, str], data_folder: str, verbose: bool) -> Tuple[int, float]:
    """Inference for one record through the tabular model (cache-first)."""
    import pandas as pd

    tabular = model_dict["tabular"]
    config = tabular["config"]
    bids = str(record[HEADERS["bids_folder"]])
    site = str(record[HEADERS["site_id"]])
    session_id = str(record[HEADERS["session_id"]])

    demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
    patient_data = load_demographics(str(demo_file), bids, record[HEADERS["session_id"]])

    idx = tabular["cache_lookup"].get(_record_key(bids, session_id))
    if idx is None:
        idx = tabular["cache_lookup"].get(bids)
    if idx is not None and tabular["cache_frame"] is not None:
        feat = tabular["cache_frame"].iloc[idx]
    else:
        base = f"{bids}_ses-{session_id}"
        raw_path = Path(data_folder) / "physiological_data" / site / f"{base}.edf"
        ann_path = Path(data_folder) / "algorithmic_annotations" / site / f"{base}_caisr_annotations.edf"
        if not raw_path.exists():
            raise ValueError(f"tabular: no raw EDF for {bids} and cache miss")
        process_record = _import_process_record()
        _rec, feat_dict, _meta = process_record((dict(patient_data), str(raw_path), str(ann_path), tabular["rename_rules"]))
        feat = pd.Series(feat_dict)

    x = assemble_record_features(feat, patient_data, config).to_frame().T
    if config.get("constant") is not None:
        p = float(config["constant"])
    else:
        if config.get("model") == "xgboost":
            import xgboost as xgb

            margin = tabular["booster"].predict(xgb.DMatrix(x.values, feature_names=config["feature_list"]))
        else:
            margin = tabular["booster"].predict(x.values)
        p = float(1.0 / (1.0 + np.exp(-margin[0])))
    threshold = float(config.get("threshold", 0.5))
    return int(p >= threshold), p
