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

import hashlib
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
_TABULAR_MODEL_PREFIX = "tabular_model_seed_"

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


def _tabular_seeds(tab: Any) -> List[int]:
    value = tab.get("seeds", None)
    seeds = [0] if value is None else [int(seed) for seed in value]
    if not seeds:
        raise ValueError("tabular: at least one training seed is required")
    return seeds


def _collect_cache_matches(train_config: Any) -> Dict[str, Any]:
    """Match demographics rows to the feature cache in canonical row order."""
    import pandas as pd

    tab = {k: v for k, v in train_config.get("tabular").items()}
    data_root = Path(train_config.db_dir)
    cache_path = str(tab.get("feature_cache", "") or "")
    cache_only_training = bool(tab.get("cache_only_training", True))
    cache_frame, cache_lookup = _load_feature_cache(cache_path)

    demo_path = data_root / DEMOGRAPHICS_FILE
    if not demo_path.is_file():
        raise FileNotFoundError(f"tabular preflight: demographics file is missing: {demo_path}")
    demo = pd.read_csv(demo_path)

    matched_entries = []
    missing_rows: List[Dict[str, Any]] = []
    for _, row in demo.iterrows():
        row = dict(row)
        bids = str(row["BidsFolder"])
        session = str(row["SessionID"])
        idx = cache_lookup.get(_record_key(bids, session))
        if idx is None:
            idx = cache_lookup.get(bids)
        if idx is not None and cache_frame is not None:
            matched_entries.append(
                (
                    (bids, session),
                    row,
                    cache_frame.iloc[idx],
                    int(bool(row.get("Cognitive_Impairment", False))),
                )
            )
        else:
            missing_rows.append(row)

    matched_entries.sort(key=lambda entry: entry[0])
    missing_rows.sort(key=lambda row: (str(row["BidsFolder"]), str(row["SessionID"])))
    rows = [entry[1] for entry in matched_entries]
    feat_series = [entry[2] for entry in matched_entries]
    labels = [entry[3] for entry in matched_entries]

    return {
        "tab": tab,
        "data_root": data_root,
        "cache_path": cache_path,
        "cache_frame": cache_frame,
        "cache_lookup": cache_lookup,
        "cache_only_training": cache_only_training,
        "demo": demo,
        "rows": rows,
        "feat_series": feat_series,
        "labels": labels,
        "missing_rows": missing_rows,
    }


def _feature_columns(columns: List[str], feature_groups: List[str]) -> List[str]:
    base = [column for column in columns if not column.startswith("meta_")]
    if feature_groups:
        base = [column for column in base if column.split("_")[0] in feature_groups]
    return base


def _record_sha256(rows: List[Dict[str, Any]], labels: List[int]) -> str:
    if len(rows) != len(labels):
        raise ValueError(f"tabular: {len(rows)} rows but {len(labels)} labels")
    entries = sorted(
        f"{_record_key(str(row['BidsFolder']), str(row['SessionID']))}\t{int(label)}" for row, label in zip(rows, labels)
    )
    return hashlib.sha256(("\n".join(entries) + "\n").encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tabular_contract(
    train_config: Any,
    inputs: Dict[str, Any],
    verbose: bool,
) -> Dict[str, Any]:
    """Validate a collected tabular training contract without fitting a model."""
    tab = inputs["tab"]
    cache_frame = inputs["cache_frame"]
    cache_only_training = inputs["cache_only_training"]
    rows = inputs["rows"]
    labels = inputs["labels"]

    if cache_only_training and not inputs["cache_path"]:
        raise RuntimeError("tabular preflight: cache_only_training=True but feature_cache is empty")
    if cache_only_training and cache_frame is None:
        raise FileNotFoundError(f"tabular preflight: feature cache is missing: {inputs['cache_path']}")

    if cache_only_training and not rows:
        raise RuntimeError("tabular preflight: no demographics rows match the feature cache")
    if cache_only_training and len(set(labels)) < 2:
        raise RuntimeError("tabular preflight: matched training rows contain only one class")

    seeds = _tabular_seeds(tab)
    model_name = str(tab.get("model", "xgboost")).lower()
    if model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"tabular preflight: unknown model {model_name!r}")

    feature_groups = list(tab.get("feature_groups") or [])
    base_feature_count = (
        len(_feature_columns([str(column) for column in cache_frame.columns], feature_groups))
        if cache_frame is not None
        else None
    )
    expected_rows = int(tab.get("expected_training_rows", 0) or 0)
    expected_features = int(tab.get("expected_feature_count", 0) or 0)
    if expected_rows and cache_only_training and len(rows) != expected_rows:
        raise RuntimeError(f"tabular preflight: expected {expected_rows} cache-matched training rows, got {len(rows)}")
    if expected_features and base_feature_count is not None and base_feature_count != expected_features:
        raise RuntimeError(
            f"tabular preflight: expected {expected_features} features for groups "
            f"{feature_groups}, got {base_feature_count}"
        )

    summary = {
        "data_root": str(inputs["data_root"]),
        "demographics_rows": int(len(inputs["demo"])),
        "cache_rows": int(len(cache_frame)) if cache_frame is not None else 0,
        "matched_rows": int(len(rows)),
        "skipped_rows": int(len(inputs["missing_rows"])),
        "positive_rows": int(np.sum(labels)) if labels else 0,
        "negative_rows": int(len(labels) - np.sum(labels)) if labels else 0,
        "base_feature_count": base_feature_count,
        "expected_training_rows": expected_rows or None,
        "expected_feature_count": expected_features or None,
        "training_record_sha256": _record_sha256(rows, labels) if rows else None,
        "feature_cache_sha256": _file_sha256(inputs["cache_path"]),
        "row_order": "BidsFolder,SessionID",
        "model": model_name,
        "seeds": seeds,
        "expected_boosters": len(seeds),
        "xgb_n_jobs": int(tab.get("xgb_n_jobs", _xgb_n_jobs())) if model_name == "xgboost" else None,
        "cache_only_training": cache_only_training,
    }
    if verbose:
        print(f"[CinC2026] tabular preflight: {summary}")
    return summary


def preflight_tabular(train_config: Any, verbose: bool = True) -> Dict[str, Any]:
    """Validate the tabular training contract without fitting a model."""
    return _validate_tabular_contract(train_config, _collect_cache_matches(train_config), verbose)


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

    inputs = _collect_cache_matches(train_config)
    _validate_tabular_contract(train_config, inputs, verbose)
    tab = inputs["tab"]
    data_root = inputs["data_root"]
    cache_frame = inputs["cache_frame"]
    cache_only_training = inputs["cache_only_training"]

    feature_groups = list(tab.get("feature_groups") or [])
    include_meta = bool(tab.get("include_meta", True))

    rows: List[Dict[str, Any]] = list(inputs["rows"])
    feat_series = list(inputs["feat_series"])
    labels: List[int] = list(inputs["labels"])
    missing_rows: List[Dict[str, Any]] = list(inputs["missing_rows"])

    if cache_only_training and cache_frame is not None:
        on_the_fly = {}
        if verbose and missing_rows:
            print(
                f"[CinC2026] tabular: cache-only training ignores {len(missing_rows)} "
                "records absent from the audited feature cache"
            )
    else:
        on_the_fly = _compute_many(data_root, missing_rows, int(tab.get("workers", 4)))
        for row in missing_rows:
            key = _record_key(str(row["BidsFolder"]), str(row["SessionID"]))
            if key in on_the_fly:
                rows.append(row)
                feat_series.append(pd.Series(on_the_fly[key]))
                labels.append(int(bool(row.get("Cognitive_Impairment", False))))
        entries = sorted(
            zip(rows, feat_series, labels),
            key=lambda entry: (str(entry[0]["BidsFolder"]), str(entry[0]["SessionID"])),
        )
        rows = [entry[0] for entry in entries]
        feat_series = [entry[1] for entry in entries]
        labels = [entry[2] for entry in entries]

    if not rows:
        raise RuntimeError("tabular: no records with features (cache empty and no raw EDFs)")
    if verbose:
        n_cache = len(feat_series) - len(on_the_fly)
        print(f"[CinC2026] tabular: {len(rows)} feature rows (cache {n_cache}, on-the-fly {len(on_the_fly)})")

    x = pd.DataFrame(feat_series)
    feature_list = _feature_columns([str(column) for column in x.columns], feature_groups)
    x = x[feature_list].reset_index(drop=True)
    expected_features = int(tab.get("expected_feature_count", 0) or 0)
    if expected_features and len(feature_list) != expected_features:
        raise RuntimeError(
            f"tabular: expected {expected_features} features for groups {feature_groups}, " f"got {len(feature_list)}"
        )
    meta_cols = ["meta_age", "meta_sex_male", "meta_bmi", "meta_rec_year"]
    if include_meta:
        meta_df = pd.DataFrame([_meta_vector(r) for r in rows], columns=meta_cols)
        x = pd.concat([x, meta_df], axis=1)
        feature_list = feature_list + meta_cols
    y = np.asarray(labels, dtype=int)

    model_name = str(tab.get("model", "xgboost")).lower()
    threshold = float(tab.get("threshold", 0.5))
    constant = None
    seeds = _tabular_seeds(tab)

    if model_name == "xgboost":
        import xgboost as xgb

        base_params = dict(tab.get("xgb_params") or {})
        base_params.pop("seed", None)
        base_params.pop("random_state", None)
        base_params.setdefault("tree_method", "hist")
        n_jobs = int(tab.get("xgb_n_jobs", _xgb_n_jobs()))
        classifiers = [xgb.XGBClassifier(random_state=seed, n_jobs=n_jobs, **base_params) for seed in seeds]
    elif model_name == "lightgbm":
        import lightgbm as lgb

        base_params = dict(tab.get("lgbm_params") or {})
        base_params.pop("seed", None)
        base_params.pop("random_state", None)
        classifiers = [lgb.LGBMClassifier(random_state=seed, **base_params) for seed in seeds]
    else:
        raise ValueError(f"tabular: unknown model {model_name!r}")

    if len(set(y.tolist())) < 2:
        constant = float(y.mean())
        if verbose:
            print(f"[CinC2026] tabular: single-class labels — constant model prob={constant:.4f}")
    else:
        for clf in classifiers:
            clf.fit(x, y)

    config = {
        "model": model_name,
        "feature_list": feature_list,
        "threshold": threshold,
        "include_meta": include_meta,
        "constant": constant,
        "data_root": str(data_root),
        "feature_cache": str(tab.get("feature_cache", "") or ""),
        "n_train": int(len(rows)),
        "training_record_sha256": _record_sha256(rows, labels),
        "feature_cache_sha256": _file_sha256(str(tab.get("feature_cache", "") or "")),
        "row_order": "BidsFolder,SessionID",
        "skipped_rows": len(missing_rows) if cache_only_training else 0,
        "base_feature_count": int(expected_features or len(feature_list) - (len(meta_cols) if include_meta else 0)),
        "seeds": seeds,
        "model_files": [],
        "cache_only_training": cache_only_training,
        "xgb_n_jobs": int(tab.get("xgb_n_jobs", _xgb_n_jobs())) if model_name == "xgboost" else None,
    }
    model_folder.mkdir(parents=True, exist_ok=True)
    if constant is None:
        for seed, clf in zip(seeds, classifiers):
            suffix = "json" if model_name == "xgboost" else "txt"
            filename = f"{_TABULAR_MODEL_PREFIX}{seed}.{suffix}"
            path = model_folder / filename
            if model_name == "xgboost":
                clf.get_booster().save_model(str(path))
            else:
                clf.booster_.save_model(str(path))
            config["model_files"].append(filename)
    if constant is None:
        assert len(config["model_files"]) == len(seeds), "tabular: seed-ensemble artifact count mismatch"
    with open(model_folder / _TABULAR_CONFIG_NAME, "w") as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"[CinC2026] tabular {len(config['model_files'])}-seed ensemble saved to {model_folder}")


def load_tabular_model(model_folder: Path, train_config: Any, verbose: bool) -> Dict[str, Any]:
    """Load the tabular artefacts into a component payload.

    The payload keys mirror what :func:`run_tabular_model` consumes; the legacy
    ``team_code.load_model`` wraps it as ``{"tabular": payload}``.
    """
    cfg_path = model_folder / _TABULAR_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"tabular config missing at {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    boosters = []
    if config.get("constant") is None:
        model_files = list(config.get("model_files") or [])
        if not model_files:
            legacy_name = _TABULAR_XGB_MODEL_NAME if config.get("model") == "xgboost" else _TABULAR_LGB_MODEL_NAME
            model_files = [legacy_name]
        if config.get("model") == "xgboost":
            import xgboost as xgb

            boosters = [xgb.Booster(model_file=str(model_folder / name)) for name in model_files]
        else:
            import lightgbm as lgb

            boosters = [lgb.Booster(model_file=str(model_folder / name)) for name in model_files]
    tab = {k: v for k, v in train_config.get("tabular").items()}
    cache_frame, cache_lookup = _load_feature_cache(tab.get("feature_cache", ""))
    rename_rules = load_rename_rules(str(Path(__file__).resolve().parent / "channel_table.csv"))
    if verbose:
        print(
            f"[CinC2026] tabular model loaded "
            f"({config.get('model')}, {len(boosters)} model(s), n_train={config.get('n_train')})"
        )
    return {
        "config": config,
        "booster": boosters[0] if len(boosters) == 1 else None,
        "boosters": boosters,
        "cache_frame": cache_frame,
        "cache_lookup": cache_lookup,
        "rename_rules": rename_rules,
    }


def run_tabular_model(
    tabular_payload: Dict[str, Any], record: Dict[str, str], data_folder: str, verbose: bool
) -> Tuple[int, float]:
    """Inference for one record through a loaded tabular payload (cache-first)."""
    import pandas as pd

    config = tabular_payload["config"]
    bids = str(record[HEADERS["bids_folder"]])
    site = str(record[HEADERS["site_id"]])
    session_id = str(record[HEADERS["session_id"]])

    demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
    patient_data = load_demographics(str(demo_file), bids, record[HEADERS["session_id"]])

    idx = tabular_payload["cache_lookup"].get(_record_key(bids, session_id))
    if idx is None:
        idx = tabular_payload["cache_lookup"].get(bids)
    if idx is not None and tabular_payload["cache_frame"] is not None:
        feat = tabular_payload["cache_frame"].iloc[idx]
    else:
        base = f"{bids}_ses-{session_id}"
        raw_path = Path(data_folder) / "physiological_data" / site / f"{base}.edf"
        ann_path = Path(data_folder) / "algorithmic_annotations" / site / f"{base}_caisr_annotations.edf"
        if not raw_path.exists():
            raise ValueError(f"tabular: no raw EDF for {bids} and cache miss")
        process_record = _import_process_record()
        _rec, feat_dict, _meta = process_record(
            (dict(patient_data), str(raw_path), str(ann_path), tabular_payload["rename_rules"])
        )
        feat = pd.Series(feat_dict)

    x = assemble_record_features(feat, patient_data, config).to_frame().T
    if config.get("constant") is not None:
        p = float(config["constant"])
    else:
        boosters = tabular_payload.get("boosters")
        if boosters is None:
            boosters = [tabular_payload.get("booster")]
        if not boosters or any(booster is None for booster in boosters):
            raise RuntimeError("tabular: no booster available for prediction")
        if config.get("model") == "xgboost":
            import xgboost as xgb

            dmat = xgb.DMatrix(x.values, feature_names=config["feature_list"])
            probabilities = [float(booster.predict(dmat)[0]) for booster in boosters]
        else:
            probabilities = [float(booster.predict(x.values)[0]) for booster in boosters]
        p = float(np.mean(probabilities))
    threshold = float(config.get("threshold", 0.5))
    return int(p >= threshold), p
