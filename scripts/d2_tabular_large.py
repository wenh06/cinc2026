#!/usr/bin/env python
"""D2-large: tabular baselines on the LARGE pool, CAISR-only + metadata.

Gate experiment for sub5: train XGBoost / LightGBM on S0001+I0002 (5,458
records) with the CAISR sleep-architecture block (``arch_*`` + ``trans_*``)
plus the four metadata columns, then evaluate the official age-conditioned
AUROC on held-out I0006 (1,138 records — 4 I0006 records lack CAISR
annotations and are skipped by the extractor).  Reference: A1 = 0.6375
(sub3-config CRNN on the large pool, I0006 holdout).

The 1.2 TB large raw PSG is not local, so the spectral/coherence/temporal/
HRV/SpO2 blocks are absent here (all-NaN, dropped column-wise).  The full
spectral bank uses the same script once the large raw is available.

Feature extraction first (CAISR-only, no raw needed)::

    python scripts/extract_spectral_features.py \
        --data-root data/official-phase-large \
        --out-dir tmp/d2_large_features \
        --workers 8 --no-require-raw

Usage::

    python scripts/d2_tabular_large.py [--seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.scoring_metrics import age_conditioned_auroc

FEATURES_CSV = PROJECT_ROOT / "tmp" / "d2_large_features" / "features.csv"
META_CSV = PROJECT_ROOT / "tmp" / "d2_large_features" / "record_meta.csv"
OUT_JSON = PROJECT_ROOT / "tmp" / "d2_large_features" / "results.json"
LARGE_CRNN_REF = 0.6375  # A1: sub3-config CRNN, large pool, I0006 holdout
META_COLS = ["meta_age", "meta_sex_male", "meta_bmi", "meta_rec_year"]

# Submission-config hyperparameters (cfg.py TrainCfg.tabular)
LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    verbosity=-1,
)
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feats = pd.read_csv(FEATURES_CSV).rename(columns={"Unnamed: 0": "record"})
    meta = pd.read_csv(META_CSV).rename(columns={"Unnamed: 0": "record"})
    assert (feats["record"].astype(str) == meta["record"].astype(str)).all()
    x = pd.concat([feats.drop(columns=["record"]), meta[META_COLS]], axis=1)
    keep = [c for c in x.columns if c in META_COLS or c.startswith(("arch_", "trans_"))]
    x = x[keep].dropna(axis=1, how="all")
    y = meta["label"].astype(int).to_numpy()
    ages = meta["meta_age"].astype(float).to_numpy()
    sites = meta["meta_site"].astype(str).to_numpy()
    return x, y, ages, sites


def score(y_true, probs, ages) -> float:
    return age_conditioned_auroc(np.asarray(probs), np.asarray(y_true), np.asarray(ages))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    x, y, ages, sites = load_data()
    train_mask = np.isin(sites, ["S0001", "I0002"])
    eval_mask = sites == "I0006"
    print(
        f"features: {x.shape[1]}  train {int(train_mask.sum())} (pos {int(y[train_mask].sum())})  "
        f"eval {int(eval_mask.sum())} (pos {int(y[eval_mask].sum())})"
    )

    results: dict[str, list[float]] = {}

    lr_meta = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    lr_meta.fit(x.loc[train_mask, META_COLS], y[train_mask])
    results["lr_meta"] = [score(y[eval_mask], lr_meta.predict_proba(x.loc[eval_mask, META_COLS])[:, 1], ages[eval_mask])]

    for seed in seeds:
        xg = xgb.XGBClassifier(random_state=seed, n_jobs=-1, **XGB_PARAMS)
        xg.fit(x.loc[train_mask], y[train_mask])
        results.setdefault("xgb_arch_meta", []).append(
            score(y[eval_mask], xg.predict_proba(x.loc[eval_mask])[:, 1], ages[eval_mask])
        )

        lm = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
        lm.fit(x.loc[train_mask], y[train_mask])
        results.setdefault("lgb_arch_meta", []).append(
            score(y[eval_mask], lm.predict_proba(x.loc[eval_mask])[:, 1], ages[eval_mask])
        )

    summary = {}
    print(f"\nreference: large-pool CRNN I0006 (A1) = {LARGE_CRNN_REF}")
    for name, vals in results.items():
        vals = np.asarray(vals)
        summary[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "seeds": [round(float(v), 4) for v in vals],
        }
        print(
            f"{name:14s} mean {vals.mean():.4f} ± {summary[name]['std']:.4f}  " f"Δ vs A1: {vals.mean() - LARGE_CRNN_REF:+.4f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "split": {"train": "S0001+I0002", "eval": "I0006"},
                "reference_large_crnn": LARGE_CRNN_REF,
                "feature_block": "arch+meta (CAISR-only; raw not local)",
                "models": summary,
            },
            f,
            indent=2,
        )
    print(f"\nresults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
