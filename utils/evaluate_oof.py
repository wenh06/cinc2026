"""evaluate_oof.py — out-of-fold evaluation of the 5-fold ensemble.

Loads the fold models trained with ``TrainCfg.folds = [0..4]``
(``model_folder/fold_{k}/final_model.pth.tar``), runs each on its own val
fold, and concatenates the predictions.  Every record is predicted by
exactly one model that never saw it during training, so the aggregate is an
unbiased CV estimate of single-model generalisation.  The official test set
instead scores the 5-model equal-weight average — OOF is the honest
(conservative) lower bound of that.

Usage::

    python utils/evaluate_oof.py -m saved_models/o5_5fold -d data/official-phase-large
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cfg import TrainCfg
from dataset import CINC2026Dataset, collate_fn
from models import EpochCRNN, EpochTransformer
from utils.misc import age_conditioned_auroc

_MODEL_CLASS_MAP = {
    "epoch_transformer": EpochTransformer,
    "epoch_transformer_S": EpochTransformer,
    "epoch_transformer_M": EpochTransformer,
    "epoch_transformer_L": EpochTransformer,
    "epoch_crnn": EpochCRNN,
    "epoch_crnn_S": EpochCRNN,
    "epoch_crnn_M": EpochCRNN,
    "epoch_crnn_L": EpochCRNN,
}

_SITES = ["S0001", "I0002", "I0006"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage::")[0].strip())
    parser.add_argument(
        "-m",
        "--model-folder",
        default="saved_models/o5_5fold",
        help="folder containing fold_*/final_model.pth.tar",
    )
    parser.add_argument("-d", "--data-dir", default="data/official-phase-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_folder = Path(args.model_folder)
    fold_dirs = sorted(model_folder.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        print(f"No fold_* dirs found under {model_folder}")
        sys.exit(1)

    model_name = TrainCfg.model_name
    model_cls = _MODEL_CLASS_MAP[model_name]

    all_probs, all_labels, all_ages, all_sites = [], [], [], []
    for fd in fold_dirs:
        fold = int(fd.name.split("_")[1])
        ckpt = torch.load(fd / "final_model.pth.tar", map_location=device, weights_only=False)
        model = model_cls(**deepcopy(ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        print(f"fold_{fold}: {sum(p.numel() for p in model.parameters()):,} params")

        cfg = deepcopy(TrainCfg)
        cfg.db_dir = str(args.data_dir)
        cfg.fold = fold
        cfg.lazy = False
        ds = CINC2026Dataset(cfg, training=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        probs, labels, ages, sites = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"fold_{fold}"):
                out = model(batch)
                probs.append(out["ci_prob"][:, 1].cpu().numpy())
                labels.append(batch["label"].cpu().numpy())
                ages.append(batch["demographics"][:, 0].cpu().numpy() * 100.0)  # age/100 → years
                sites.extend(batch["site_id"])

        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        ages = np.concatenate(ages)
        sites = np.asarray(sites)
        if len(labels) < 2 or len(np.unique(labels)) < 2:
            print(f"  fold_{fold}: only {len(labels)} samples / single class — skipping metrics")
        else:
            auroc = roc_auc_score(labels, probs)
            auroc_ac = age_conditioned_auroc(probs, labels, ages)
            print(f"  fold_{fold} val: AUROC={auroc:.4f}  age-cond={auroc_ac:.4f}  (n={len(labels)})")

        all_probs.append(probs)
        all_labels.append(labels)
        all_ages.append(ages)
        all_sites.append(sites)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ages = np.concatenate(all_ages)
    sites = np.concatenate(all_sites)

    print("\n" + "=" * 60)
    print(f"OOF aggregate ({len(labels)} records, one prediction per record)")
    print(f"  AUROC          : {roc_auc_score(labels, probs):.4f}")
    print(f"  age-cond AUROC : {age_conditioned_auroc(probs, labels, ages):.4f}")
    for site in _SITES:
        mask = sites == site
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            continue
        print(
            f"  {site:<6s} (n={mask.sum():5d}): AUROC={roc_auc_score(labels[mask], probs[mask]):.4f}  "
            f"age-cond={age_conditioned_auroc(probs[mask], labels[mask], ages[mask]):.4f}"
        )


if __name__ == "__main__":
    main()
