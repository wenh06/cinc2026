"""make_5fold_split.py — generate the 5-fold CV split for ensemble training.

Layout: for each fold k, ``{"train": [...], "val": [...]}`` with the val
folds non-overlapping and covering the whole labelled set, and fold k's
train set = all records not in fold k's val set (≈ 80% of the data).
Training fold k then validates on ``fold_k.val``; at inference the 5 fold
models' probabilities are averaged (equal-weight ensemble).

Stratification is multi-factor — label × site × sex × age band — via
torch_ecg's ``stratified_train_test_split`` (the same function the dataset
uses for its dynamic split), so every fold mirrors the population on each
measured demographic axis.  The folds are carved recursively:

    fold_0.val ← stratified split of all          data, test_ratio = 1/5
    fold_1.val ← stratified split of the remainder, test_ratio = 1/4
    fold_2.val ← stratified split of the remainder, test_ratio = 1/3
    fold_3.val ← stratified split of the remainder, test_ratio = 1/2
    fold_4.val ← the final remainder

Seeded via ``DEFAULTS.RNG`` (torch_ecg, seed 42) → reproducible.

Usage::

    python utils/make_5fold_split.py -d data/official-phase-large
    python utils/make_5fold_split.py -d data/official-phase-large --out /tmp/5fold.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from torch_ecg.utils.utils_data import stratified_train_test_split

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from const import FIVE_FOLD_SPLIT_FILE  # noqa: E402

# Age banding mirrors dataset.py's dynamic split (5-year bands over 50-90).
AGE_BINS = [49, 55, 60, 65, 70, 75, 80, 85, 100]
AGE_LABELS = ["50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-90"]
STRAT_COLS = ["Cognitive_Impairment", "SiteID", "Sex", "AgeGroup"]

_TRAIN_PARTS = ("training_set", "training_set_small", "training_set_large")


def load_labelled_df(db_dir: Path) -> pd.DataFrame:
    """Build the labelled-records dataframe (mirrors CINC2026Dataset)."""
    dfs: List[pd.DataFrame] = []
    for part in _TRAIN_PARTS:
        demo = db_dir / part / "demographics.csv"
        if demo.exists():
            df = pd.read_csv(demo)
            df["partition"] = part
            dfs.append(df)
    if not dfs:
        flat = db_dir / "demographics.csv"
        if flat.exists():
            dfs.append(pd.read_csv(flat))
    if not dfs:
        raise FileNotFoundError(f"No demographics.csv found under {db_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df["BidsFolder"] = df["BidsFolder"].astype(str)
    return df.set_index("BidsFolder")


def make_folds(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Carve *n_splits* non-overlapping stratified val folds.

    Returns ``{"fold_k": {"train": [...], "val": [...]}}`` where every fold's
    val set is disjoint from the others, their union is the full dataset, and
    ``fold_k.train`` = full dataset minus ``fold_k.val``.
    """
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    remaining = df
    all_index = set(df.index)
    val_sets: List[set] = []
    for k in range(n_splits):
        if k == n_splits - 1:
            val_sets.append(set(remaining.index))
            continue
        _, val_df = stratified_train_test_split(
            remaining,
            stratified_cols=STRAT_COLS,
            test_ratio=1.0 / (n_splits - k),
        )
        val_sets.append(set(val_df.index))
        remaining = remaining.loc[~remaining.index.isin(val_df.index)]

    folds = {}
    for k in range(n_splits):
        val = sorted(val_sets[k])
        train = sorted(all_index - val_sets[k])
        folds[f"fold_{k}"] = {"train": train, "val": val}
    return folds


def report_balance(df: pd.DataFrame, folds: dict, strat_cols: Sequence[str]) -> None:
    """Print per-fold val distributions vs the full population, per stratum."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    for col in strat_cols:
        full = df[col].value_counts(normalize=True).sort_index()
        print(f"\n{col}:")
        for k in range(len(folds)):
            val = df.loc[df.index.isin(folds[f"fold_{k}"]["val"]), col].value_counts(normalize=True).sort_index()
            dev = (val - full).abs().max()
            print(f"  fold_{k}: max |val - full| = {dev * 100:5.2f}%  (n={len(folds[f'fold_{k}']['val'])})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage::")[0].strip())
    parser.add_argument(
        "-d",
        "--db-dir",
        default="data/official-phase-large",
        help="data root containing demographics.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output json path (default: the repo-shipped FIVE_FOLD_SPLIT_FILE)",
    )
    args = parser.parse_args()

    df = load_labelled_df(Path(args.db_dir))
    print(f"Loaded {len(df)} labelled records from {args.db_dir}")

    folds = make_folds(df)
    report_balance(df, folds, STRAT_COLS)

    out_path = Path(args.out) if args.out else Path(FIVE_FOLD_SPLIT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"\nSaved 5-fold split to {out_path}")


if __name__ == "__main__":
    main()
