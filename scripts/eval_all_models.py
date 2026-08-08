"""Evaluate one or more trained models on the FULL training data through the
official evaluation flow (``run_model`` inference chain + official metrics).

Mirrors the organisers' evaluation as closely as the training data allows:
``find_patients`` → per-record ``run_model`` (sliding-window inference,
demographics handling, tuned binary threshold / ensemble majority vote) →
the official metric set (AUROC / age-conditioned AUROC / age-weighted AUROC /
AUPRC / Accuracy / F-measure / reward), plus a per-site breakdown.

⚠️ The records are training data (seen by the model), so the absolute numbers
are optimistic.  The point is a FAIR, identical-pipeline comparison across
models / configs (e.g. O0repro vs O5 vs O7 variants) on the same records.
Prevalence is computed from the full training demographics (official
``compute_prevalence``, age gap = 2).

Usage::

    python scripts/eval_all_models.py -d data/official-phase-large \\
        -m saved_models/official_baseline \\
        -m saved_models/o5_5fold \\
        -m saved_models/o7_pairwise \\
        -o results/o7_compare.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper_code import find_patients  # official helper (unmodified)
from team_code import load_model, run_model
from utils.scoring_metrics import age_conditioned_auroc, age_weighted_auroc, compute_reward

_METRIC_ORDER = [
    "auroc",
    "auroc_age_cond",
    "auroc_age_weighted",
    "auprc",
    "reward",
    "accuracy",
    "f_measure",
]


def _compute_prevalence(labels: np.ndarray, ages: np.ndarray) -> Dict[float, float]:
    """Official ``compute_prevalence`` (age gap = 2 years) over the training set."""
    unique_ages = np.unique(ages[np.isfinite(ages)])
    age_to_prevalence: Dict[float, float] = {}
    for a in unique_ages:
        m = np.abs(ages - a) <= 2.0
        age_to_prevalence[float(a)] = max(float(labels[m].sum()), 0.5) / m.sum()
    return age_to_prevalence


def _evaluate(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    ages: np.ndarray,
    site_ids: np.ndarray,
    age_to_prevalence: Dict[float, float],
    no_site: bool,
) -> Dict[str, float]:
    """Full official metric set (+ per-site breakdown).

    AUROC-family metrics use the (averaged / sliding-window) probabilities;
    Accuracy / F1 / reward use the *binary* predictions exactly as produced
    by ``run_model`` (tuned per-fold thresholds, majority vote for the
    5-fold ensemble) — the same split of responsibilities as the official
    ``evaluate_model.py``.
    """
    metrics = {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "auroc_age_cond": age_conditioned_auroc(probs, labels, ages),
        "auroc_age_weighted": age_weighted_auroc(probs, labels, ages),
        "accuracy": accuracy_score(labels, preds),
        "f_measure": f1_score(labels, preds),
        "reward": compute_reward(labels, preds, ages, age_to_prevalence),
    }
    if not no_site:
        for site in np.unique(site_ids):
            m = site_ids == site
            if len(np.unique(labels[m])) < 2:
                continue
            metrics[f"auroc_{site}"] = roc_auc_score(labels[m], probs[m])
            metrics[f"auroc_age_cond_{site}"] = age_conditioned_auroc(probs[m], labels[m], ages[m])
            metrics[f"reward_{site}"] = compute_reward(labels[m], preds[m], ages[m], age_to_prevalence)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on the full training data via the official flow")
    parser.add_argument(
        "-d",
        "--db-dir",
        required=True,
        help="data partition root (demographics.csv + algorithmic_annotations/)",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        dest="models",
        required=True,
        help="model folder (repeatable; single model or fold_*/ ensemble layout)",
    )
    parser.add_argument("-o", "--output", default=None, help="write the results table to a file")
    parser.add_argument("--device", default="cuda", help="inference device (cuda | cpu)")
    parser.add_argument("--no-site", action="store_true", help="skip the per-site breakdown")
    args = parser.parse_args()

    # Route the inference device into team_code's module-level DEVICE before
    # load_model / run_model use it.
    if args.device != "cuda":
        import torch

        import team_code

        team_code.DEVICE = torch.device(args.device)

    db_dir = Path(args.db_dir)
    demo_file = db_dir / "demographics.csv"
    if not demo_file.exists():
        raise FileNotFoundError(f"no demographics.csv in {db_dir}")

    # ── Records + metadata (all of them — this is the training set) ──────
    records = find_patients(str(demo_file))
    df = pd.read_csv(demo_file)
    if not df["BidsFolder"].is_unique:
        raise ValueError("demographics.csv has duplicate BidsFolder rows")
    meta = df.set_index("BidsFolder")
    n = len(records)
    labels = np.zeros(n)
    ages = np.zeros(n)
    site_ids = np.empty(n, dtype=object)
    for i, rec in enumerate(records):
        row = meta.loc[rec["BidsFolder"]]
        labels[i] = int(row["Cognitive_Impairment"])
        ages[i] = float(row["Age"])
        site_ids[i] = row["SiteID"]

    # ── Prevalence from the full training demographics (official, gap=2) ──
    age_to_prevalence = _compute_prevalence(
        df["Cognitive_Impairment"].to_numpy(dtype=float),
        df["Age"].to_numpy(dtype=float),
    )

    print("=" * 92)
    print("  Model evaluation on the FULL training set — official flow")
    print(f"  data     : {db_dir}   ({n} records, {int(labels.sum())} CI-positive)")
    print("=" * 92)

    all_results: Dict[str, Dict[str, float]] = {}
    for model_path in args.models:
        label = Path(model_path).name
        print(f"\n--- {label} ---")
        model_dict = load_model(str(model_path), verbose=True)
        probs = np.zeros(n)
        preds = np.zeros(n, dtype=int)
        for i, rec in enumerate(tqdm(records, desc=f"  [{label}]", unit="rec")):
            pred, prob = run_model(model_dict, rec, str(db_dir), verbose=False)
            probs[i] = prob
            preds[i] = int(pred)
        all_results[label] = _evaluate(labels, probs, preds, ages, site_ids, age_to_prevalence, args.no_site)

    # ── Summary table ──────────────────────────────────────────────────────
    # Column width fits the longest metric name (e.g. ``auroc_age_weighted``),
    # so the header doesn't run into the next column.
    col_width = max(len(k) for k in _METRIC_ORDER) + 2
    lines = ["=" * 92, "  SUMMARY — official metrics on the full training set", "=" * 92]
    lines.append(f"{'model':<28s}" + "".join(f"{k:>{col_width}s}" for k in _METRIC_ORDER))
    for label, metrics in all_results.items():
        lines.append(f"{label:<28s}" + "".join(f"{metrics.get(k, float('nan')):>{col_width}.4f}" for k in _METRIC_ORDER))
    if not args.no_site:
        lines.append("")
        for site in sorted(np.unique(site_ids)):
            lines.append(f"--- per-site {site} ---")
            lines.append(f"{'model':<28s}{'auroc':>12s}{'auroc_age_cond':>18s}{'reward':>12s}")
            for label, metrics in all_results.items():
                lines.append(
                    f"{label:<28s}"
                    f"{metrics.get(f'auroc_{site}', float('nan')):>12.4f}"
                    f"{metrics.get(f'auroc_age_cond_{site}', float('nan')):>18.4f}"
                    f"{metrics.get(f'reward_{site}', float('nan')):>12.4f}"
                )

    text = "\n".join(lines)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
