#!/usr/bin/env python
"""D2: tabular baselines on small-set spectral features (I0006-holdout proxy).

Trains GBDT/LR on the 641-dim spectral/CAISR feature bank (``tmp/spectral_features/``)
plus record metadata, evaluates the official age-conditioned AUROC on the held-out
site I0006 (train = S0001 + I0002, 911 records; eval = I0006, 192 records / 20 pos).

References:
* **0.546** — B1: sub3-config CRNN trained on the SMALL pool, I0006 holdout
  (the comparable number for this data regime; A1's 0.6375 uses the LARGE pool,
  which is gated on the 1.2 TB large-raw download).
* Discipline: report per-seed runs; adopt only seed-robust Δ > 0.04.

Usage::

    python scripts/d2_tabular_baseline.py [--seeds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.scoring_metrics import age_conditioned_auroc

FEATURES_CSV = PROJECT_ROOT / "tmp" / "spectral_features" / "features.csv"
META_CSV = PROJECT_ROOT / "tmp" / "spectral_features" / "record_meta.csv"
OUT_ROOT = PROJECT_ROOT / "tmp" / "d2_tabular"
SMALL_CRNN_REF = 0.546  # B1 small-pool, I0006 holdout
META_COLS = ["meta_age", "meta_sex_male", "meta_bmi", "meta_rec_year"]


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feats = pd.read_csv(FEATURES_CSV).rename(columns={"Unnamed: 0": "record"})
    meta = pd.read_csv(META_CSV).rename(columns={"Unnamed: 0": "record"})
    assert (feats["record"].astype(str) == meta["record"].astype(str)).all()
    x = pd.concat(
        [feats.drop(columns=["record"]), meta[META_COLS]],
        axis=1,
    )
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
    train_mask = sites != "I0006"
    eval_mask = sites == "I0006"
    spectral_cols = [c for c in x.columns if c not in META_COLS]
    print(
        f"train {int(train_mask.sum())} (pos {int(y[train_mask].sum())}), "
        f"eval {int(eval_mask.sum())} (pos {int(y[eval_mask].sum())})"
    )
    print(f"all-NaN feature rows: {(x[spectral_cols].isna().all(axis=1)).sum()}")

    results: dict[str, list[float]] = {}

    # meta-only floors
    lr_meta = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    lr_meta.fit(x.loc[train_mask, META_COLS], y[train_mask])
    p = lr_meta.predict_proba(x.loc[eval_mask, META_COLS])[:, 1]
    results["lr_meta"] = [score(y[eval_mask], p, ages[eval_mask])]
    for seed in seeds:
        m = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
        )
        m.fit(x.loc[train_mask, META_COLS], y[train_mask])
        p = m.predict_proba(x.loc[eval_mask, META_COLS])[:, 1]
        results.setdefault("lgb_meta", []).append(score(y[eval_mask], p, ages[eval_mask]))

    # full feature models
    lr_full = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=0.1),
    )
    lr_full.fit(x.loc[train_mask], y[train_mask])
    p = lr_full.predict_proba(x.loc[eval_mask])[:, 1]
    results["lr_full"] = [score(y[eval_mask], p, ages[eval_mask])]
    for seed in seeds:
        m = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            verbosity=-1,
        )
        m.fit(x.loc[train_mask], y[train_mask])
        p = m.predict_proba(x.loc[eval_mask])[:, 1]
        results.setdefault("lgb_full", []).append(score(y[eval_mask], p, ages[eval_mask]))
        m.fit(x.loc[train_mask, spectral_cols], y[train_mask])
        p = m.predict_proba(x.loc[eval_mask, spectral_cols])[:, 1]
        results.setdefault("lgb_spectral", []).append(score(y[eval_mask], p, ages[eval_mask]))

    summary = {}
    print(f"\nreference: CRNN small-pool I0006 = {SMALL_CRNN_REF}")
    for name, vals in results.items():
        vals = np.asarray(vals)
        summary[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "seeds": [round(float(v), 4) for v in vals],
        }
        extra = f"  Δ vs {SMALL_CRNN_REF}: {vals.mean() - SMALL_CRNN_REF:+.4f}" if name != "lr_meta" else ""
        print(f"{name:14s} mean {vals.mean():.4f} ± {summary[name]['std']:.4f}  {extra}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "results.json", "w") as f:
        json.dump(
            {
                "split": {"train": "S0001+I0002", "eval": "I0006"},
                "reference_small_crnn": SMALL_CRNN_REF,
                "models": summary,
            },
            f,
            indent=2,
        )
    print(f"\nresults saved to {OUT_ROOT / 'results.json'}")


if __name__ == "__main__":
    main()
