from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch_ecg.utils.misc import make_serializable

from helper_code import compute_accuracy, compute_auc, compute_challenge_score, compute_f_measure
from outputs import CINC2026Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, torch.Tensor]]],
    outputs: Sequence[CINC2026Outputs],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute the challenge metrics for CinC 2026 Cognitive Impairment prediction.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, torch.Tensor]]]
        The labels produced by the dataset class, expected key is "labels".
    outputs : Sequence[CINC2026Outputs]
        The outputs produced by the model.
    """
    if not outputs or not all(hasattr(o, "ci_prob") and o.ci_prob is not None for o in outputs):
        return {m: np.nan for m in ["challenge_score", "auroc", "auprc", "accuracy", "f_measure", "tpr"]}

    # 1. Extract and concatenate Ground Truth labels
    all_labels = []
    for label_dict in labels:
        lb = label_dict["labels"]
        if isinstance(lb, torch.Tensor):
            lb = lb.cpu().detach().numpy()
        # Handle scalar labels vs arrays
        all_labels.append(np.atleast_1d(lb))
    all_labels = np.concatenate(all_labels)

    # Convert from one-hot if necessary
    if all_labels.ndim > 1:
        all_labels = np.argmax(all_labels, axis=1)

    # 2. Extract and concatenate model outputs
    # ci_prob[:, 1] is the probability of the positive (impaired) class
    all_probs = np.concatenate([np.asarray(o.ci_prob)[:, 1] for o in outputs])
    all_preds = np.concatenate([np.asarray(o.cognitive_impairment) for o in outputs])

    # 3. Evaluate using helper_code logic
    challenge_score = compute_challenge_score(all_labels, all_probs)
    auroc, auprc = compute_auc(all_labels, all_probs)
    accuracy = compute_accuracy(all_labels, all_preds)
    f_measure = compute_f_measure(all_labels, all_preds)

    # Custom TPR computation
    tpr = compute_ci_tpr(all_labels, all_preds)

    results = {
        "challenge_score": challenge_score,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "f_measure": f_measure,
        "tpr": tpr,
    }

    if verbose:
        print("\n" + "-" * 30)
        print("CinC 2026 Evaluation Results")
        for k, v in results.items():
            print(f"{k:15s}: {v:.4f}")
        print("-" * 30)

    return make_serializable(results)


def compute_ci_tpr(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute True Positive Rate (Recall / Sensitivity)."""
    tp = np.sum((labels == 1) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    return float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
