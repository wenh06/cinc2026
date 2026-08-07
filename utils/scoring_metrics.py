"""Official scoring metrics and the binary-threshold tuner."""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import CINC2026Dataset, collate_fn

__all__ = [
    "age_conditioned_auroc",
    "age_weighted_auroc",
    "compute_challenge_metrics",
    "compute_reward",
    "tune_binary_threshold",
]


def age_conditioned_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    age_tolerance: float = 2.0,
) -> float:
    """Official primary metric: AUROC over positive-negative pairs whose ages
    differ by at most ``age_tolerance`` years.  Returns 0.5 (chance) when only
    one class is present or no valid age-matched pair exists.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    ages = np.asarray(ages, dtype=np.float64)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.5

    n_agree = 0.0
    n_pairs = 0
    for i in pos_idx:
        age_diff = np.abs(ages[neg_idx] - ages[i])
        valid = neg_idx[age_diff <= age_tolerance]
        if len(valid) == 0:
            continue
        p_pos = probs[i]
        p_neg = probs[valid]
        n_agree += float((p_pos > p_neg).sum()) + 0.5 * float((p_pos == p_neg).sum())
        n_pairs += len(valid)

    if n_pairs == 0:
        return 0.5
    return float(n_agree / n_pairs)


def age_weighted_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    age_tolerance: float = 2.0,
) -> float:
    """Official metric: age-weighted mean of per-age-window AUROC.

    Equivalent to the official ``compute_auroc_weighted`` (a window is a
    year value in [min_age - tol, max_age + tol]; its AUROC is over the
    records within ±tol of it, weighted by the window's record count).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    ages = np.asarray(ages, dtype=np.float64)

    finite = np.isfinite(ages)
    if not finite.any():
        return 0.5
    range_ages = np.arange(ages[finite].min() - age_tolerance, ages[finite].max() + age_tolerance + 1)

    weights, aucs = [], []
    for a in range_ages:
        win = np.abs(ages - a) <= age_tolerance
        if win.sum() < 2 or len(np.unique(labels[win])) < 2:
            continue
        weights.append(win.sum())
        aucs.append(roc_auc_score(labels[win], probs[win]))

    if not weights:
        return 0.5
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return float(np.sum(weights * np.asarray(aucs)))


def compute_reward(
    labels: np.ndarray,
    preds: np.ndarray,
    ages: np.ndarray,
    age_to_prevalence: Dict[float, float],
) -> float:
    """Official secondary metric: prevalence-weighted accuracy.

    True positive scores ``1/p - 1`` (≈ 12 at p=0.076), FP/FN score ``-1``,
    true negative scores ``1/(1-p) - 1``; equivalently ``(TP/p - FP/(1-p))/n``.
    """
    n = len(labels)
    scores = np.zeros(n)
    num_scores = 0
    for i in range(n):
        if np.isfinite(ages[i]):
            p = age_to_prevalence[ages[i]]
            p = min(max(p, 0.5 / n), 1 - 0.5 / n)
            if labels[i] == 1 and preds[i] == 1:
                scores[i] = 1 / p - 1
            elif labels[i] == 1 and preds[i] == 0:
                scores[i] = -1
            elif labels[i] == 0 and preds[i] == 1:
                scores[i] = -1
            elif labels[i] == 0 and preds[i] == 0:
                scores[i] = 1 / (1 - p) - 1
            num_scores += 1
    return float(np.sum(scores) / num_scores)


def compute_challenge_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    ages: np.ndarray,
    site_ids: Optional[np.ndarray] = None,
    age_to_prevalence: Optional[Dict[float, float]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the official metric set in one call.

    Returns ``auroc``, ``auprc``, ``auroc_age_cond``, ``auroc_age_weighted``,
    ``accuracy``, ``f_measure``; ``reward`` only when ``age_to_prevalence`` is
    given; per-site ``auroc_<site>`` / ``auroc_age_cond_<site>`` when
    ``site_ids`` is given.  ``auroc_age_cond`` falls back to plain AUROC when
    no valid age-matched pair exists.  Binary metrics use ``threshold``.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    ages = np.asarray(ages)
    preds = (probs >= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        metrics = {
            "auroc": 0.5,
            "auprc": float(np.mean(labels)) if len(labels) else 0.5,
            "auroc_age_cond": 0.5,
            "auroc_age_weighted": 0.5,
            "accuracy": float(np.mean(preds == labels)),
            "f_measure": 0.0,
        }
    else:
        auroc = float(roc_auc_score(labels, probs))
        auroc_age_cond = age_conditioned_auroc(probs, labels, ages, age_tolerance=2.0)
        if auroc_age_cond == 0.5 and auroc != 0.5:
            auroc_age_cond = auroc
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        f_measure = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics = {
            "auroc": auroc,
            "auprc": float(average_precision_score(labels, probs)),
            "auroc_age_cond": auroc_age_cond,
            "auroc_age_weighted": age_weighted_auroc(probs, labels, ages, age_tolerance=2.0),
            "accuracy": float(np.mean(preds == labels)),
            "f_measure": f_measure,
        }

    if age_to_prevalence is not None:
        metrics["reward"] = compute_reward(labels, preds, ages, age_to_prevalence)

    if site_ids is not None:
        site_arr = np.asarray(site_ids)
        for site in sorted(np.unique(site_arr)):
            m = site_arr == site
            if m.sum() > 1 and len(np.unique(labels[m])) > 1:
                metrics[f"auroc_{site}"] = float(roc_auc_score(labels[m], probs[m]))
                metrics[f"auroc_age_cond_{site}"] = age_conditioned_auroc(probs[m], labels[m], ages[m])

    return metrics


def tune_binary_threshold(
    model: torch.nn.Module,
    dataset: CINC2026Dataset,
    device: torch.device,
) -> float:
    """Scan thresholds on *dataset* and return the one maximizing Reward.

    Prevalence is estimated from the dataset reader's training records
    (official ``compute_prevalence``, gap=2).  Called with the best
    checkpoint; the returned threshold is stored in the model config so
    ``run_model`` reproduces it.  Affects Reward/Accuracy/F1 only.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    probs, labels, ages = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            probs.append(out["ci_prob"][:, 1].cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            ages.append(batch["demographics"][:, 0].cpu().numpy() * 100.0)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    ages = np.concatenate(ages)

    rec_df = dataset.reader._df_records
    train_ages = rec_df["Age"].astype(float).values
    train_labels = rec_df["Cognitive_Impairment"].astype(int).values

    # Official compute_prevalence: keyed on the evaluation ages, ±2-year window
    unique_ages = np.unique(ages[np.isfinite(ages)])
    age_to_prevalence = {}
    for age in unique_ages:
        m = np.abs(train_ages - age) <= 2.0
        age_to_prevalence[age] = max(train_labels[m].sum(), 0.5) / m.sum()

    best_thr, best_reward = 0.5, -np.inf
    for t in np.arange(0.02, 0.55, 0.01):
        preds = (probs >= t).astype(int)
        r = compute_reward(labels, preds, ages, age_to_prevalence)
        if r > best_reward:
            best_reward, best_thr = r, t
    print(f"  [tune_binary_threshold] best reward {best_reward:+.4f} @ threshold " f"{best_thr:.2f} (n={len(labels)})")
    return float(best_thr)
