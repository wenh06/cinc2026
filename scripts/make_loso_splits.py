"""Generate deterministic leave-one-site-out (LOSO) splits.

The generated splits are keyed by held-out site and have the same ``train`` /
``val`` layout consumed by :class:`dataset.CINC2026Dataset`.  A manifest stores
the record-content hash and file SHA-256 for each split so evaluation scripts
can reject stale or manually edited files.

Usage::

    python scripts/make_loso_splits.py
    python scripts/make_loso_splits.py -d data/official-phase-large --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "official-phase-large"
DEFAULT_OUT_DIR = PROJECT_ROOT / "scripts" / "splits" / "loso"
DEFAULT_SITES = ("I0002", "I0006", "S0001")
TRAIN_PARTS = ("training_set", "training_set_small", "training_set_large")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def record_digest(train: Iterable[str], val: Iterable[str]) -> str:
    """Stable digest of the record assignment, independent of JSON formatting."""
    payload = "train\n" + "\n".join(sorted(map(str, train))) + "\nval\n" + "\n".join(sorted(map(str, val)))
    return _sha256_bytes(payload.encode("utf-8"))


def load_labeled_records(db_dir: Path) -> pd.DataFrame:
    """Load labelled records using the same partition names as the dataset."""
    frames: list[pd.DataFrame] = []
    for part in TRAIN_PARTS:
        path = db_dir / part / "demographics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["partition"] = part
            frames.append(frame)

    flat_path = db_dir / "demographics.csv"
    if not frames and flat_path.exists():
        frame = pd.read_csv(flat_path)
        frame["partition"] = "training_set"
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No demographics.csv found under {db_dir}")

    frame = pd.concat(frames, ignore_index=True)
    required = {"BidsFolder", "SiteID", "Cognitive_Impairment"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"demographics.csv missing columns: {sorted(missing)}")

    frame = frame.loc[frame["Cognitive_Impairment"].notna()].copy()
    frame["BidsFolder"] = frame["BidsFolder"].astype(str)
    duplicates = frame["BidsFolder"][frame["BidsFolder"].duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"duplicate BidsFolder values: {duplicates[:10].tolist()}")

    def annotation_path(row: pd.Series) -> str:
        partition = str(row["partition"])
        data_root = db_dir / partition if (db_dir / partition).is_dir() else db_dir
        base = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        return str(data_root / "algorithmic_annotations" / str(row["SiteID"]) / f"{base}_caisr_annotations.edf")

    frame["algo_ann_path"] = frame.apply(annotation_path, axis=1)
    frame["caisr_available"] = frame["algo_ann_path"].map(lambda value: Path(value).is_file())
    return frame.set_index("BidsFolder", drop=False)


def data_digest(frame: pd.DataFrame) -> str:
    columns = [c for c in ("BidsFolder", "SiteID", "SessionID", "Cognitive_Impairment", "caisr_available") if c in frame]
    rows = frame.loc[:, columns].sort_index().astype(str)
    payload = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def build_splits(
    frame: pd.DataFrame,
    sites: Iterable[str] = DEFAULT_SITES,
) -> dict[str, dict]:
    """Build one exact train/validation split per held-out site."""
    all_records = set(frame.index.astype(str))
    site_to_records = {site: set(frame.index[frame["SiteID"].astype(str) == site].astype(str)) for site in sites}
    if any(not records for records in site_to_records.values()):
        empty = [site for site, records in site_to_records.items() if not records]
        raise ValueError(f"no records found for sites: {empty}")

    splits: dict[str, dict] = {}
    for held_out, val_records in site_to_records.items():
        train_records = all_records.difference(val_records)
        overlap = train_records.intersection(val_records)
        if overlap:
            raise AssertionError(f"{held_out}: train/val overlap: {sorted(overlap)[:10]}")
        if train_records.union(val_records) != all_records:
            raise AssertionError(f"{held_out}: split does not cover the labelled dataset")

        train = sorted(train_records)
        val = sorted(val_records)
        train_sites = sorted(frame.loc[train, "SiteID"].astype(str).unique())
        val_sites = sorted(frame.loc[val, "SiteID"].astype(str).unique())
        train_usable = sorted(frame.index[frame["caisr_available"] & frame.index.isin(train)].astype(str))
        val_usable = sorted(frame.index[frame["caisr_available"] & frame.index.isin(val)].astype(str))
        if held_out in train_sites or val_sites != [held_out]:
            raise AssertionError(f"{held_out}: invalid site assignment; train={train_sites}, val={val_sites}")

        splits[held_out] = {
            "schema_version": 1,
            "protocol": "leave_one_site_out",
            "held_out_site": held_out,
            "train_sites": train_sites,
            "val_sites": val_sites,
            "counts": {
                "total": len(all_records),
                "train": len(train),
                "val": len(val),
                "train_positive": int(frame.loc[train, "Cognitive_Impairment"].astype(int).sum()),
                "val_positive": int(frame.loc[val, "Cognitive_Impairment"].astype(int).sum()),
                "train_caisr_available": len(train_usable),
                "val_caisr_available": len(val_usable),
                "train_positive_caisr_available": int(frame.loc[train_usable, "Cognitive_Impairment"].astype(int).sum()),
                "val_positive_caisr_available": int(frame.loc[val_usable, "Cognitive_Impairment"].astype(int).sum()),
            },
            "record_sha256": record_digest(train, val),
            "train": train,
            "val": val,
        }
    return splits


def _serialize(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_splits(
    frame: pd.DataFrame,
    out_dir: Path,
    sites: Iterable[str] = DEFAULT_SITES,
    check: bool = False,
) -> None:
    splits = build_splits(frame, sites=sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, dict] = {}
    stale: list[str] = []

    for site, split in splits.items():
        path = out_dir / f"split_{site}.json"
        payload = _serialize(split)
        if check:
            if not path.exists() or path.read_bytes() != payload:
                stale.append(str(path))
        else:
            path.write_bytes(payload)
        manifest_splits[site] = {
            "path": path.name,
            "sha256": _sha256_bytes(payload),
            "record_sha256": split["record_sha256"],
            "counts": split["counts"],
            "train_sites": split["train_sites"],
            "val_sites": split["val_sites"],
        }

    manifest = {
        "schema_version": 1,
        "protocol": "leave_one_site_out",
        "data_sha256": data_digest(frame),
        "total_records": len(frame),
        "splits": manifest_splits,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_payload = _serialize(manifest)
    if check:
        if not manifest_path.exists() or manifest_path.read_bytes() != manifest_payload:
            stale.append(str(manifest_path))
        if stale:
            raise SystemExit("LOSO split files are missing or stale:\n" + "\n".join(stale))
        print(f"LOSO splits are current: {out_dir}")
    else:
        manifest_path.write_bytes(manifest_payload)
        print(f"Wrote {len(splits)} LOSO splits and manifest to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that generated content matches the committed files without writing",
    )
    args = parser.parse_args()

    frame = load_labeled_records(args.data_dir.expanduser().resolve())
    write_splits(frame, args.out_dir.expanduser().resolve(), sites=args.sites, check=args.check)


if __name__ == "__main__":
    main()
