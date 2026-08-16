#!/usr/bin/env python
"""D2 robustness: cross-family checks + feature-group attribution.

Confirms the first D2 reading (LightGBM on the 641-dim spectral bank:
0.671±0.018, I0006-holdout of the small set) with
* more seeds for LightGBM,
* XGBoost / RandomForest as cross-model-family checks,
* single-feature-group models (which block carries the signal; ``arch`` is the
  21-dim CAISR-derived block closest to the CRNN input).

Usage::

    python scripts/d2_tabular_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from d2_tabular_baseline import META_COLS, SMALL_CRNN_REF, load_data, score

OUT_ROOT = PROJECT_ROOT / "tmp" / "d2_tabular"
GROUPS = ["spec", "coh", "tp", "trans", "arch", "hrv", "spo2"]


def make_lgb(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed,
        verbosity=-1,
    )


def make_xgb(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)


def main() -> None:
    x, y, ages, sites = load_data()
    train_mask = sites != "I0006"
    eval_mask = sites == "I0006"
    spectral_cols = [c for c in x.columns if c not in META_COLS]

    results: dict[str, dict] = {}

    def run_block(name: str, cols: list[str], seeds: range, factory) -> None:
        vals = []
        for seed in seeds:
            m = factory(seed)
            m.fit(x.loc[train_mask, cols], y[train_mask])
            p = m.predict_proba(x.loc[eval_mask, cols])[:, 1]
            vals.append(score(y[eval_mask], p, ages[eval_mask]))
        vals = np.asarray(vals)
        results[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "seeds": [round(float(v), 4) for v in vals],
        }

    run_block("lgb_spectral", spectral_cols, range(15), make_lgb)
    run_block("xgb_spectral", spectral_cols, range(5), make_xgb)
    run_block("rf_spectral", spectral_cols, range(5), make_rf)
    for g in GROUPS:
        cols = [c for c in spectral_cols if c.startswith(f"{g}_")]
        run_block(f"lgb_{g}_only", cols, range(5), make_lgb)

    print(f"\nreference: CRNN small-pool I0006 = {SMALL_CRNN_REF}")
    for name, r in results.items():
        delta = r["mean"] - SMALL_CRNN_REF
        print(f"{name:18s} mean {r['mean']:.4f} ± {r['std']:.4f} " f"[{r['min']:.4f}, {r['max']:.4f}]  Δ {delta:+.4f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "robustness.json", "w") as f:
        json.dump(
            {
                "split": {"train": "S0001+I0002", "eval": "I0006"},
                "reference_small_crnn": SMALL_CRNN_REF,
                "models": results,
            },
            f,
            indent=2,
        )
    print(f"\nresults saved to {OUT_ROOT / 'robustness.json'}")


if __name__ == "__main__":
    main()
