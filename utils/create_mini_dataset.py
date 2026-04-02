"""Create a mini training subset for GitHub CI testing.

Selects a balanced set of records (CAISR EDFs + demographics only) and
packages them as a zip archive.  Records are sampled proportionally from the
canonical train and validation splits (``utils/cinc2026-data-split.json``) so
that when ``CINC2026Dataset`` runs on the mini dataset with
``override_data_split=False`` (the default), the same train/val assignments
are recovered automatically.

Usage::

    python create_mini_dataset.py \\
        --db-dir /Data1/wenh06/physionetchallenge2026data \\
        --out-dir /tmp/cinc2026_mini_build \\
        --zip /Data1/wenh06/cinc2026-mini-training-set.zip \\
        --n-per-stratum 30

The output zip contains::

    training_set/
        demographics.csv
        algorithmic_annotations/
            S0001/<record>_ses-<session>_caisr_annotations.edf
            I0002/...
            I0006/...

To use this in GitHub CI, upload the zip to a publicly accessible URL
(e.g., Google Drive) and set that URL as ``MINI_DATASET_URL`` in
``.github/workflows/docker-test.yml``, then change ``status`` from ``pre``
to ``alpha``.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

SITES = ["S0001", "I0002", "I0006"]
DEMOGRAPHICS_FILE = "demographics.csv"
ANN_SUBDIR = "algorithmic_annotations"
_PROJECT_DIR = Path(__file__).resolve().parent
_CANONICAL_SPLIT_FILE = _PROJECT_DIR / "cinc2026-data-split.json"
_TRAIN_RATIO = 0.8  # must match StratifiedShuffleSplit train_size in dataset.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", required=True, help="Root data directory (parent of training_set/)")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Scratch directory for assembling the subset (will be deleted and recreated)",
    )
    p.add_argument(
        "--zip",
        required=True,
        help="Destination path for the output zip archive (e.g. /Data1/wenh06/cinc2026-mini-training-set.zip)",
    )
    p.add_argument(
        "--n-per-stratum",
        type=int,
        default=30,
        help="Total records per (SiteID × CI label) stratum; sampled ~80%% from canonical train, ~20%% from val",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    db_dir = Path(args.db_dir)
    train_dir = db_dir / "training_set"
    out_root = Path(args.out_dir)
    out_train = out_root / "training_set"
    out_ann = out_train / ANN_SUBDIR

    if not train_dir.exists():
        print(f"ERROR: training_set not found at {train_dir}", file=sys.stderr)
        sys.exit(1)

    demo_path = train_dir / DEMOGRAPHICS_FILE
    df = pd.read_csv(demo_path)
    df = df.set_index("BidsFolder", drop=False)

    # ------------------------------------------------------------------
    # Load canonical split to decide which records belong to train vs val
    # ------------------------------------------------------------------
    if not _CANONICAL_SPLIT_FILE.exists():
        print(f"ERROR: canonical split file not found at {_CANONICAL_SPLIT_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(_CANONICAL_SPLIT_FILE) as f:
        canonical = json.load(f)
    canonical_train = set(canonical.get("train", []))
    canonical_val = set(canonical.get("val", []))

    n_total = args.n_per_stratum
    # Proportional allocation: round train_n up so fractions stay consistent
    n_from_train = min(n_total, math.ceil(n_total * _TRAIN_RATIO))
    n_from_val = n_total - n_from_train

    # ------------------------------------------------------------------
    # Sample records per site × CI stratum
    # ------------------------------------------------------------------
    selected_rows = []
    for site in SITES:
        for ci_val in [True, False]:
            stratum_mask = (df["SiteID"] == site) & (df["Cognitive_Impairment"] == ci_val)
            stratum = df[stratum_mask]

            train_pool = stratum[stratum["BidsFolder"].isin(canonical_train)]
            val_pool = stratum[stratum["BidsFolder"].isin(canonical_val)]

            n_t = min(n_from_train, len(train_pool))
            n_v = min(n_from_val, len(val_pool))

            sampled = pd.concat(
                [
                    train_pool.sample(n=n_t, random_state=args.seed) if n_t > 0 else pd.DataFrame(),
                    val_pool.sample(n=n_v, random_state=args.seed) if n_v > 0 else pd.DataFrame(),
                ],
                ignore_index=True,
            )
            if len(sampled) == 0:
                print(f"  WARNING: no records for site={site} CI={ci_val}, skipping stratum")
                continue
            selected_rows.append(sampled)
            print(
                f"  {site} CI={ci_val}: {n_t} from train + {n_v} from val "
                f"= {len(sampled)} (pool: train={len(train_pool)}, val={len(val_pool)})"
            )

    selected_df = pd.concat(selected_rows, ignore_index=True)
    print(f"\nTotal selected: {len(selected_df)} records")

    # ------------------------------------------------------------------
    # Build output directory structure
    # ------------------------------------------------------------------
    if out_root.exists():
        shutil.rmtree(out_root)
    out_ann.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        (out_ann / site).mkdir(exist_ok=True)

    # Copy CAISR EDFs
    src_ann = train_dir / ANN_SUBDIR
    copied = 0
    missing = []
    for _, row in selected_df.iterrows():
        site = row["SiteID"]
        bids_folder = row["BidsFolder"]
        session_id = int(row["SessionID"])
        edf_name = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"
        src_edf = src_ann / site / edf_name
        dst_edf = out_ann / site / edf_name
        if src_edf.exists():
            shutil.copy2(src_edf, dst_edf)
            copied += 1
        else:
            missing.append(str(src_edf))

    if missing:
        print(f"\nWARNING: {len(missing)} CAISR EDFs not found:")
        for m in missing:
            print(f"  {m}")

    print(f"Copied {copied} CAISR EDF files.")

    # Write filtered demographics.csv (only rows whose CAISR EDF was found)
    found_bids = {
        row["BidsFolder"]
        for _, row in selected_df.iterrows()
        if (src_ann / row["SiteID"] / f"{row['BidsFolder']}_ses-{int(row['SessionID'])}_caisr_annotations.edf").exists()
    }
    out_df = selected_df[selected_df["BidsFolder"].isin(found_bids)].drop(columns=["BidsFolder"], errors="ignore")
    # Re-read from original to get all columns in the right order
    orig_df = pd.read_csv(demo_path)
    out_df = orig_df[orig_df["BidsFolder"].isin(found_bids)]
    out_df.to_csv(out_train / DEMOGRAPHICS_FILE, index=False)
    print(f"Wrote demographics.csv with {len(out_df)} rows.")

    # ------------------------------------------------------------------
    # Package as zip
    # ------------------------------------------------------------------
    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(out_root), base_dir="training_set")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\nCreated {zip_path} ({size_mb:.1f} MB)")
    print("Upload this file to a publicly accessible URL and set MINI_DATASET_URL")
    print("in .github/workflows/docker-test.yml, then set status to 'alpha'.")

    # Clean up scratch directory
    shutil.rmtree(out_root)
    print("Scratch directory cleaned up.")


if __name__ == "__main__":
    main()
