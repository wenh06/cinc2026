#!/usr/bin/env python
"""Fetch algorithmic_annotations from the PhysioNet Challenge 2026 dataset.

Usage::

    python scripts/fetch_official_phase.py                          # small, default dir
    python scripts/fetch_official_phase.py --size large              # large, default dir
    python scripts/fetch_official_phase.py --size large -d ./my_dir  # custom dir
    python scripts/fetch_official_phase.py --list-only               # just list files

Does two things:

1. Lists all files in the Kaggle dataset via ``kaggle datasets files``
   and writes the full listing to ``all_files.txt`` (if not already present).
2. Downloads every ``algorithmic_annotations/`` file via ``kagglehub``
   into ``<data_dir>/algorithmic_annotations/``, preserving the internal
   directory structure.  Already-downloaded files are skipped automatically.

Requires: ``kaggle`` CLI with valid ``~/.kaggle/kaggle.json`` credentials,
and ``kagglehub>=1.0`` (the ``output_dir`` parameter was added in 1.x).
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ANNO_PREFIX = "algorithmic_annotations/"

SIZE_CONFIG = {
    "small": {
        "dataset": "physionet/physionetchallenge2026data",
        "default_dir": "data/official-phase",
    },
    "large": {
        "dataset": "physionet/physionetchallenge2026datalargeversion",
        "default_dir": "data/official-phase-large",
    },
}

# ---------------------------------------------------------------------------
# Step 1 — List dataset files via kaggle CLI
# ---------------------------------------------------------------------------


def list_all_files(dataset: str, data_dir: Path) -> Path:
    """Fetch the full file listing from Kaggle, cache as *data_dir*/all_files.txt."""
    output = data_dir / "all_files.txt"
    if output.exists():
        print(f"[list] {output} already exists — skipping (delete it to re-fetch)")
        return output

    print(f"[list] Fetching file listing for {dataset} ...")
    token = ""
    with output.open("w") as fh:
        while True:
            cmd = ["kaggle", "datasets", "files", dataset, "--page-size", "200"]
            if token:
                cmd += ["--page-token", token]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[list] ERROR: {result.stderr.strip()}")
                sys.exit(1)

            lines = result.stdout.splitlines()
            for line in lines[3:]:
                if line.startswith("Next Page Token"):
                    token = line.split("=", 1)[-1].strip()
                    break
                parts = line.split()
                if parts and not parts[0].startswith("-"):
                    fh.write(parts[0] + "\n")
            else:
                break

    n = sum(1 for _ in output.open())
    print(f"[list] Wrote {n} file paths to {output}")
    return output


# ---------------------------------------------------------------------------
# Step 2 — Filter algorithmic_annotations
# ---------------------------------------------------------------------------


def filter_anno_paths(all_files: Path, data_dir: Path) -> List[str]:
    """Return sorted list of ``algorithmic_annotations/`` paths."""
    anno_file = data_dir / "anno.txt"
    if anno_file.exists():
        paths = [line.strip() for line in anno_file.read_text().splitlines() if line.strip()]
        print(f"[filter] Loaded {len(paths)} paths from {anno_file}")
        return paths

    paths = sorted(line.strip() for line in all_files.read_text().splitlines() if line.strip().startswith(ANNO_PREFIX))
    anno_file.write_text("\n".join(paths) + "\n")
    print(f"[filter] Wrote {len(paths)} paths to {anno_file}")
    return paths


# ---------------------------------------------------------------------------
# Step 3 — Download via kagglehub
# ---------------------------------------------------------------------------


def download_files(dataset: str, paths: List[str], data_dir: Path, max_retries: int = 3) -> None:
    """Download files to *data_dir*/, preserving the ``algorithmic_annotations/`` tree."""
    import kagglehub  # imported here so --help works without the dep

    ok = fail = 0
    t0 = time.time()

    for i, rel_path in enumerate(paths, 1):
        err = None
        for attempt in range(1, max_retries + 1):
            try:
                kagglehub.dataset_download(dataset, path=rel_path, output_dir=str(data_dir))
                break
            except Exception as exc:
                err = str(exc)
                if attempt < max_retries:
                    time.sleep(2**attempt)

        if err is not None:
            fail += 1
            print(f"[{i}/{len(paths)}] FAIL {rel_path}: {err}")
        else:
            ok += 1

        if i % 200 == 0 or i == len(paths):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(paths) - i) if i < len(paths) else 0
            print(f"  Progress: {i}/{len(paths)}  " f"OK={ok}  FAIL={fail}  " f"elapsed={elapsed/60:.0f}m  ETA={eta/60:.0f}m")

    # Clean up kagglehub completion markers
    complete = data_dir / ".complete"
    if complete.exists():
        shutil.rmtree(complete)

    print(f"Done. {ok} OK, {fail} failed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch algorithmic_annotations from CinC 2026 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-s",
        "--size",
        choices=["small", "large"],
        default="small",
        help="Dataset variant (default: small)",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=None,
        help="Target directory (default: data/official-phase[-large])",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list files (skip download)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip file listing (assume anno.txt already exists)",
    )
    args = parser.parse_args()

    cfg = SIZE_CONFIG[args.size]
    dataset = cfg["dataset"]
    data_dir = (args.data_dir or Path(cfg["default_dir"])).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] size={args.size}  dataset={dataset}  data_dir={data_dir}")

    if not args.download_only:
        all_files = list_all_files(dataset, data_dir)
        paths = filter_anno_paths(all_files, data_dir)
    else:
        paths = filter_anno_paths(data_dir / "all_files.txt", data_dir)

    if args.list_only:
        print(f"[list] {len(paths)} algorithmic_annotations files listed. Done.")
        return

    download_files(dataset, paths, data_dir)


if __name__ == "__main__":
    main()
