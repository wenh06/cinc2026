"""Verify committed LOSO splits against the labelled dataset and manifest.

This check is intentionally dependency-light so it can run on the host or in
the official container before any model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from make_loso_splits import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_SITES,
    build_splits,
    data_digest,
    load_labeled_records,
    record_digest,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-s", "--split-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    frame = load_labeled_records(args.data_dir.expanduser().resolve())
    expected = build_splits(frame, sites=DEFAULT_SITES)
    manifest_path = args.split_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    current_data_sha = data_digest(frame)
    if manifest.get("data_sha256") != current_data_sha:
        raise SystemExit(
            "manifest data digest does not match the current labelled data; " "labels or CAISR availability changed"
        )
    if int(manifest.get("total_records", -1)) != len(frame):
        raise SystemExit(f"manifest total_records={manifest.get('total_records')} does not match current {len(frame)}")

    for site in DEFAULT_SITES:
        path = args.split_dir / f"split_{site}.json"
        if not path.exists():
            raise SystemExit(f"missing split: {path}")
        split = json.loads(path.read_text())
        current = expected[site]

        if split["train"] != current["train"] or split["val"] != current["val"]:
            raise SystemExit(f"{site}: committed split does not match labelled dataset")
        if record_digest(split["train"], split["val"]) != split["record_sha256"]:
            raise SystemExit(f"{site}: record digest is invalid")
        if set(split["train"]).intersection(split["val"]):
            raise SystemExit(f"{site}: train/val overlap")
        if site in split["train_sites"] or split["val_sites"] != [site]:
            raise SystemExit(f"{site}: held-out site occurs in the wrong split")

        entry = manifest["splits"][site]
        if entry["sha256"] != sha256(path):
            raise SystemExit(f"{site}: file SHA does not match manifest")
        if entry["record_sha256"] != split["record_sha256"]:
            raise SystemExit(f"{site}: record SHA does not match manifest")
        if entry["counts"] != split["counts"]:
            raise SystemExit(f"{site}: CAISR/raw counts do not match the current data")

        print(
            f"[OK] {site}: train={len(split['train'])} "
            f"({','.join(split['train_sites'])}), "
            f"val={len(split['val'])} ({site}); "
            f"CAISR available={entry['counts']['train_caisr_available']}/"
            f"{entry['counts']['val_caisr_available']} (train/val)"
        )

    print(f"[OK] LOSO manifest and all {len(DEFAULT_SITES)} splits are reproducible")


if __name__ == "__main__":
    main()
