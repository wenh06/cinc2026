"""Scheduler A/B on the LO-site proxy: sub3 config, one change — the LR schedule.

  one_cycle    : current default (max_lr 1e-3, pct_start 0.3)
  warmup_cosine: 2023+ community default — linear 5%-warmup to base lr 3e-4,
                 then cosine decay to 0 (SequentialLR, trainer override)

Baseline for comparison: A1 0.6375 (sub3 config, one_cycle, I0006-holdout).

Usage: python lo_site_sched.py I0006 warmup_cosine [--seed 0]
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset as dataset_mod
from cfg import TrainCfg
from data_reader import CINC2026
from dataset import CINC2026Dataset, collate_fn
from models import EpochCRNN
from team_code import FINAL_MODEL_NAME, _train_single_fold
from utils.scoring_metrics import compute_challenge_metrics

DATA = str(Path(__file__).resolve().parents[1] / "data" / "official-phase-large")
OUT_ROOT = Path("/tmp/lo_site")
SPLIT_ROOT = Path(__file__).resolve().parent / "splits" / "loso"
TRAIN_PARTS = {"training_set", "training_set_small", "training_set_large"}


def _record_digest(train, val) -> str:
    payload = "train\n" + "\n".join(sorted(train)) + "\nval\n" + "\n".join(sorted(val))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_and_validate_split(path: Path, held_out_site: str, frame):
    if not path.exists():
        raise FileNotFoundError(f"committed LOSO split is missing: {path}; " "run scripts/make_loso_splits.py first")
    with open(path) as f:
        split = json.load(f)

    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LOSO split manifest is missing: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest_entry = manifest.get("splits", {}).get(held_out_site, {})
    if manifest_entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"{path} does not match its manifest SHA-256")

    if split.get("protocol") != "leave_one_site_out":
        raise ValueError(f"{path} is not a LOSO split")
    if split.get("held_out_site") != held_out_site:
        raise ValueError(f"{path} holds out {split.get('held_out_site')}, expected {held_out_site}")

    train, val = split["train"], split["val"]
    expected_train = set(frame.index[frame.SiteID != held_out_site])
    expected_val = set(frame.index[frame.SiteID == held_out_site])
    if set(train) != expected_train or set(val) != expected_val:
        raise ValueError(f"{path} does not match SiteID partitioning in {DATA}")
    if set(train).intersection(val):
        raise ValueError(f"{path} has train/validation overlap")
    if split.get("record_sha256") != _record_digest(train, val):
        raise ValueError(f"{path} record digest mismatch")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_site")
    ap.add_argument("sched", choices=["one_cycle", "warmup_cosine"])
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=100, help="training epochs (smoke: 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test_site = args.test_site
    np.random.seed(args.seed)

    reader = CINC2026(db_dir=DATA)
    df = reader._df_records[reader._df_records["partition"].isin(TRAIN_PARTS)].copy()
    split_file = SPLIT_ROOT / f"split_{test_site}.json"
    _load_and_validate_split(split_file, test_site, df)
    dataset_mod.FIXED_DATA_SPLIT_FILE = str(split_file)

    cfg = deepcopy(TrainCfg)
    cfg.db_dir = DATA
    cfg.strict_data_split = True
    cfg.folds = None
    cfg.n_epochs = args.epochs
    cfg.lr_scheduler = args.sched
    cfg.warmup_frac = args.warmup_frac

    tag = f"sched_{args.sched}"
    out = OUT_ROOT / f"train_{test_site}_{tag}"
    _train_single_fold(cfg, out, verbose=True)

    # ---- evaluate on the held-out site ----
    ckpt = torch.load(out / FINAL_MODEL_NAME, map_location="cuda", weights_only=False)
    model = EpochCRNN(**deepcopy(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to("cuda")
    model.eval()

    cfg_eval = deepcopy(cfg)
    cfg_eval.lazy = False
    ds = CINC2026Dataset(cfg_eval, training=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
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

    train_ages = df[df.SiteID != test_site]["Age"].astype(float).values
    train_labels = df[df.SiteID != test_site]["Cognitive_Impairment"].astype(int).values
    unique_ages = np.unique(ages[np.isfinite(ages)])
    age_to_prevalence = {}
    for a in unique_ages:
        m = np.abs(train_ages - a) <= 2.0
        age_to_prevalence[a] = max(train_labels[m].sum(), 0.5) / m.sum()

    metrics = compute_challenge_metrics(
        labels,
        probs,
        ages,
        age_to_prevalence=age_to_prevalence,
        threshold=float(ckpt["model_config"].get("binary_threshold", 0.5)),
    )
    print(
        f"[LO-site {test_site} {tag}] age_cond={metrics['auroc_age_cond']:.4f} "
        f"auroc={metrics['auroc']:.4f} reward={metrics['reward']:.4f}"
    )
    with open(OUT_ROOT / f"metrics_{test_site}_{tag}.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
