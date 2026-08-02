"""
Post-hoc probability calibration for CinC 2026 (P2, Phase 10.2).

**Analysis utility only** — this script does NOT change the submission
pipeline, and the fitted parameters are not meant to be applied at inference
in official submissions.

Rationale: temperature scaling and Platt scaling are monotonic transforms of
the raw logits, so they leave every rank-based metric — plain AUROC and
age-conditioned AUROC alike — *exactly* unchanged.  Their only effect is on
calibration quality (Brier score, expected calibration error), which the
official leaderboard does not rank on.  This script quantifies how
miscalibrated a trained model is and what the correction ceiling would be,
for write-up purposes and for any threshold-dependent decision (e.g.
picking an operating point).

Fits on the validation split only (same split used for training
monitoring); the fitted temperature / Platt coefficients are reported but
never folded back into the model.

Usage:
    python scripts/calibrate.py \
        -d data/official-phase-large \
        -m saved_models/official_baseline/BestModel_EpochCRNN-epoch58_08-02_11-47_metric_0.77.pth.tar \
        -o saved_models/official_baseline/calibration_report.json
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from cfg import TrainCfg  # noqa: E402
from dataset import CINC2026Dataset, collate_fn  # noqa: E402
from models import EpochCRNN  # noqa: E402
from utils.misc import age_conditioned_auroc  # noqa: E402

N_BINS = 10  # ECE bins


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc calibration analysis for CinC 2026")
    parser.add_argument("-d", "--data-dir", type=str, default="data/official-phase-large", dest="data_dir")
    parser.add_argument("-s", "--split-file", type=str, default="utils/cinc2026-data-split.json", dest="split_file")
    parser.add_argument("-m", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-o", "--output-json", type=str, default="calibration_report.json", dest="output_json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    return parser.parse_args()


def load_model(model_path: str, device: str) -> EpochCRNN:
    """Load a best-model checkpoint (same loader as scripts/diagnose_age.py)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_config = deepcopy(ckpt["model_config"])
    model = EpochCRNN(**model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device) -> dict:
    """Collect raw logits, probs, labels, ages over the validation set."""
    logits, probs, labels, ages = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            output = model(batch)
            logits.append(output["ci_logits"][:, 0].cpu().numpy())
            probs.append(output["ci_prob"][:, 1].cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            ages.append(batch["demographics"][:, 0].cpu().numpy() * 100.0)  # age/100 → years
    return {
        "logits": np.concatenate(logits),
        "probs": np.concatenate(probs),
        "labels": np.concatenate(labels).astype(np.float64),
        "ages": np.concatenate(ages),
    }


def binary_nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of the temperature-scaled model.

    ``scipy.optimize.minimize_scalar`` calls this as ``f(x, *args)``, so the
    temperature must be the first positional parameter.
    """
    z = torch.from_numpy(logits / temperature)
    y = torch.from_numpy(labels)
    return float(torch.nn.functional.binary_cross_entropy_with_logits(z, y).numpy())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error (equal-width bins, sample-weighted)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(bins[1:-1], probs, side="right"), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(float(probs[mask].mean() - labels[mask].mean()))
    return total / len(probs)


def summarize(name: str, probs: np.ndarray, logits: np.ndarray, labels: np.ndarray, ages: np.ndarray) -> dict:
    """Metrics for one probability source (raw / temperature / Platt)."""
    auroc = roc_auc_score(labels, probs)
    age_cond = age_conditioned_auroc(probs, labels, ages, age_tolerance=2.0)
    brier = brier_score_loss(labels, probs)
    e = ece(probs, labels)
    corr_age = float(np.corrcoef(ages, probs)[0, 1])
    out = {
        "source": name,
        "auroc": auroc,
        "age_cond_auroc": age_cond,
        "brier": brier,
        "ece": e,
        "mean_prob": float(probs.mean()),
        "r_age_prob": corr_age,
    }
    print(
        f"  {name:<14s} AUROC={auroc:.4f}  age-cond={age_cond:.4f}  "
        f"Brier={brier:.4f}  ECE={e:.4f}  mean_p={probs.mean():.4f}  r(age,p)={corr_age:+.4f}"
    )
    return out


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    # Val-only dataset (same construction as diagnose_age.py).
    val_config = deepcopy(TrainCfg)
    val_config.db_dir = str(data_dir)
    val_config.lazy = False
    val_dataset = CINC2026Dataset(val_config, training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )
    print(f"Val subjects: {len(val_dataset)}")

    model = load_model(str(model_path), args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    d = run_inference(model, val_loader, args.device)
    labels, ages = d["labels"], d["ages"]
    prevalence = float(labels.mean())
    print(f"Prevalence: {prevalence:.4f}")

    # ── Temperature scaling (single scalar, NLL-minimising) ────────────────
    res = minimize_scalar(binary_nll, args=(d["logits"], labels), bounds=(0.05, 10.0), method="bounded")
    temperature = float(res.x)
    print(f"\nTemperature scaling: T={temperature:.4f} (val NLL {res.fun:.4f})")
    temp_probs = 1.0 / (1.0 + np.exp(-d["logits"] / temperature))

    # ── Platt scaling (affine on logits, fitted by logistic regression) ────
    platt = LogisticRegression(C=1e5, max_iter=1000).fit(d["logits"].reshape(-1, 1), labels)
    platt_a, platt_b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    print(f"Platt scaling: a={platt_a:.4f}, b={platt_b:.4f}")
    platt_probs = platt.predict_proba(d["logits"].reshape(-1, 1))[:, 1]

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CALIBRATION SUMMARY (val split)")
    print("=" * 78)
    print(f"  {'source':<14s} {'AUROC':>7s} {'age-cond':>9s} {'Brier':>7s} {'ECE':>7s} {'mean_p':>8s} {'r(age,p)':>9s}")
    results = [
        summarize("raw", d["probs"], d["logits"], labels, ages),
        summarize("temperature", temp_probs, d["logits"] / temperature, labels, ages),
        summarize("platt", platt_probs, d["logits"], labels, ages),
    ]
    print(f"\n  Prevalence = {prevalence:.4f}; AUROC is rank-based → unchanged by both methods (by construction).")
    print("  These parameters are for analysis only; do NOT fold into the submission pipeline.")
    print("=" * 78)

    report = {
        "model_path": str(model_path),
        "n_val": int(len(labels)),
        "prevalence": prevalence,
        "temperature": temperature,
        "platt": {"a": platt_a, "b": platt_b},
        "results": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
