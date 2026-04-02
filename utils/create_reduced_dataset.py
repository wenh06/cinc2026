"""Create a reduced training dataset for GitHub CI.

Copies **all** records from the training_set but includes only the files that
the model actually consumes:

- ``demographics.csv``
- ``algorithmic_annotations/`` (CAISR EDF files, ≈ 378 MB total)

Excluded (not used by the model):

- ``physiological_data/`` (raw PSG signals, ≈ 160 GB)
- ``human_annotations/`` (manual sleep-stage labels, ≈ 197 MB)

Usage::

    python create_reduced_dataset.py \\
        --db-dir /Data1/wenh06/physionetchallenge2026data \\
        --zip /Data1/wenh06/cinc2026-reduced-training-set.zip

The output zip contains::

    training_set/
        demographics.csv
        algorithmic_annotations/
            S0001/<record>_ses-<session>_caisr_annotations.edf
            I0002/...
            I0006/...

Upload the resulting zip to a publicly accessible host (Google Drive,
HuggingFace, etc.) and update ``REDUCED_DATASET_GDRIVE_ID`` (or equivalent)
in ``.github/workflows/docker-test.yml``.
"""

import argparse
import shutil
import sys
from pathlib import Path

SITES = ["S0001", "I0002", "I0006"]
DEMOGRAPHICS_FILE = "demographics.csv"
ANN_SUBDIR = "algorithmic_annotations"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", required=True, help="Root data directory (parent of training_set/)")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Scratch directory for assembling the subset (default: <db-dir>/../cinc2026_reduced_build). "
        "Deleted and recreated on each run.",
    )
    p.add_argument(
        "--zip",
        required=True,
        help="Destination path for the output zip archive " "(e.g. /Data1/wenh06/cinc2026-reduced-training-set.zip)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    db_dir = Path(args.db_dir).resolve()
    train_dir = db_dir / "training_set"

    if not train_dir.exists():
        print(f"ERROR: training_set not found at {train_dir}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir).resolve() if args.out_dir else db_dir.parent / "cinc2026_reduced_build"
    out_train = out_root / "training_set"
    out_ann = out_train / ANN_SUBDIR

    # -----------------------------------------------------------------------
    # Build output directory
    # -----------------------------------------------------------------------
    if out_root.exists():
        shutil.rmtree(out_root)
    out_ann.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        (out_ann / site).mkdir(exist_ok=True)

    # Copy demographics.csv
    src_demo = train_dir / DEMOGRAPHICS_FILE
    shutil.copy2(src_demo, out_train / DEMOGRAPHICS_FILE)
    print(f"Copied {DEMOGRAPHICS_FILE}")

    # Copy all CAISR EDFs site by site
    src_ann = train_dir / ANN_SUBDIR
    total_copied = 0
    total_missing = 0
    for site in SITES:
        src_site = src_ann / site
        dst_site = out_ann / site
        if not src_site.exists():
            print(f"  WARNING: site directory not found: {src_site}")
            continue
        edfs = sorted(src_site.glob("*_caisr_annotations.edf"))
        for edf in edfs:
            shutil.copy2(edf, dst_site / edf.name)
            total_copied += 1
        print(f"  {site}: copied {len(edfs)} CAISR EDFs")

    print(f"\nTotal CAISR EDFs copied: {total_copied}")
    if total_missing:
        print(f"WARNING: {total_missing} EDFs were expected but not found.")

    # -----------------------------------------------------------------------
    # Package as zip
    # -----------------------------------------------------------------------
    zip_path = Path(args.zip).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(out_root), base_dir="training_set")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\nCreated {zip_path}  ({size_mb:.1f} MB)")
    print(
        "\nNext steps:\n"
        "  1. Upload the zip to a publicly accessible host (Google Drive, HuggingFace, …)\n"
        "  2. Set REDUCED_DATASET_GDRIVE_ID (or equivalent) in .github/workflows/docker-test.yml\n"
        "  3. Update the download step to use this new ID instead of MINI_DATASET_GDRIVE_ID"
    )

    shutil.rmtree(out_root)
    print("Scratch directory cleaned up.")


if __name__ == "__main__":
    main()
