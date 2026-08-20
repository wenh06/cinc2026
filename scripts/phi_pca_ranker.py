#!/usr/bin/env python
"""P4: frozen Philosopher's Stone latents -> PCA-64 -> tabular ranker.

Evaluates the downstream plan from ROADMAP P4: load the precomputed 1024-D Phi
latents (+ the four brain-health scores) from the cache produced by
``scripts/phi_cache_extract.py``, PCA them to ``--pca-dim`` components, and fit
an XGBoost / logistic-regression ranker on the small-pool I0006 holdout — the
official age-conditioned AUROC proxy.  Reference: small-pool CRNN 0.546; the
large-pool A1 0.6375 is printed for context but is a different data regime.

Usage::

    python scripts/phi_pca_ranker.py \
        --cache-dir tmp/phi_cache \
        --data-root /Data1/wenh06/physionetchallenge2026data \
        --seeds 0,1,2,3,4 [--pca-dim 64] [--scores]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.phi_cache import PHI_SCORE_KEYS, load_phi_cache
from utils.scoring_metrics import age_conditioned_auroc

SMALL_CRNN_REF = 0.546
LARGE_CRNN_REF = 0.6375  # A1, context only (large train pool)
OUT_JSON = PROJECT_ROOT / "tmp" / "phi_cache" / "results_pca.json"

# submission-config XGBoost hyperparameters (cfg.py TrainCfg.tabular)
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
)


def score(y_true, probs, ages) -> float:
    return age_conditioned_auroc(np.asarray(probs), np.asarray(y_true), np.asarray(ages))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "tmp" / "phi_cache"))
    parser.add_argument("--data-root", default="/Data1/wenh06/physionetchallenge2026data")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument(
        "--holdout-site",
        choices=["I0006", "S0001", "I0002"],
        default="I0006",
        help="site left out for evaluation; the other two sites train",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="output JSON path (default: tmp/phi_cache/results_pca_<site>.json)",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="append the four brain-health scores to the PCA features",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    demo = pd.read_csv(Path(args.data_root) / "demographics.csv")
    lat = load_phi_cache(args.cache_dir, demo)
    cached_mask = ~lat.isna().all(axis=1)
    n_cached = int(cached_mask.sum())
    print(f"cached records: {n_cached}/{len(demo)}")
    if n_cached < 10:
        print("  too few cached records — smoke only, numbers are NOT meaningful")

    demo = demo.loc[cached_mask].reset_index(drop=True)
    latent_cols = [c for c in lat.columns if c.startswith("lhl_")]
    score_cols = [f"phi_{k}" for k in PHI_SCORE_KEYS]
    x = lat.loc[cached_mask, latent_cols].reset_index(drop=True)
    x_scores = lat.loc[cached_mask, score_cols].reset_index(drop=True)
    y = demo["Cognitive_Impairment"].astype(int).to_numpy()
    ages = demo["Age"].astype(float).to_numpy()
    sites = demo["SiteID"].astype(str).to_numpy()
    holdout = args.holdout_site
    eval_mask = sites == holdout
    train_mask = ~eval_mask
    if int(eval_mask.sum()) == 0:
        print(f"  no cached {holdout} records yet — nothing to score; exiting (cache still filling).")
        return
    print(
        f"train {int(train_mask.sum())} (pos {int(y[train_mask].sum())}), "
        f"eval {int(eval_mask.sum())} (pos {int(y[eval_mask].sum())})"
    )

    pca = PCA(
        n_components=min(args.pca_dim, x.shape[1], int(train_mask.sum())),
        random_state=0,
    )
    z_train = pca.fit_transform(x.loc[train_mask].to_numpy())
    z_eval = pca.transform(x.loc[eval_mask].to_numpy())
    if args.scores:
        z_train = np.hstack([z_train, x_scores.loc[train_mask].to_numpy()])
        z_eval = np.hstack([z_eval, x_scores.loc[eval_mask].to_numpy()])
    print(f"feature dim after PCA: {z_train.shape[1]} (explained var {pca.explained_variance_ratio_.sum():.3f})")

    results: dict[str, list[float]] = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(z_train, y[train_mask])
    results["lr_pca"] = [score(y[eval_mask], lr.predict_proba(z_eval)[:, 1], ages[eval_mask])]

    for seed in seeds:
        # env-overridable thread count; default -1 (all cores), set
        # CINC2026_XGB_N_JOBS=2 on the shared box (n_jobs=-1 thrashes there)
        clf = xgb.XGBClassifier(
            random_state=seed,
            n_jobs=int(os.environ.get("CINC2026_XGB_N_JOBS", "-1")),
            **XGB_PARAMS,
        )
        clf.fit(z_train, y[train_mask])
        results.setdefault("xgb_pca", []).append(score(y[eval_mask], clf.predict_proba(z_eval)[:, 1], ages[eval_mask]))

    summary = {}
    print(f"\nreference: small-pool CRNN I0006 = {SMALL_CRNN_REF}  (large-pool A1 = {LARGE_CRNN_REF})")
    for name, vals in results.items():
        vals = np.asarray(vals)
        summary[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "seeds": [round(float(v), 4) for v in vals],
        }
        print(
            f"{name:12s} mean {vals.mean():.4f} ± {summary[name]['std']:.4f}  Δ vs small-CRNN: {vals.mean() - SMALL_CRNN_REF:+.4f}"
        )

    out_json = Path(args.out_json) if args.out_json else PROJECT_ROOT / "tmp" / "phi_cache" / f"results_pca_{holdout}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "split": {"train": "+".join(sorted(set(sites[train_mask]))), "eval": holdout},
                "n_cached": n_cached,
                "pca_dim": int(z_train.shape[1] - (len(score_cols) if args.scores else 0)),
                "include_scores": bool(args.scores),
                "reference_small_crnn": SMALL_CRNN_REF,
                "reference_large_crnn": LARGE_CRNN_REF,
                "models": summary,
            },
            f,
            indent=2,
        )
    print(f"\nresults saved to {out_json}")


if __name__ == "__main__":
    main()
