#!/usr/bin/env python
"""Precompute Philosopher's Stone latents for the challenge training set.

Intended for a GPU machine (e.g. an AutoDL RTX 5090 instance); ~2-4 min/record
on a consumer card, much faster on datacenter GPUs.  Produces the cache layout
consumed by ``utils/phi_cache.py``.

Notes
-----
* The model intentionally runs in train mode (per-sample batch-norm updates),
  so outputs depend mildly on processing order; records are processed in a
  fixed (SiteID, BidsFolder) order and resume skips finished files.
* C4-M1 is taken directly when present, otherwise derived as C4 - M1 (I0006
  monopolar), falling back to bare C4.
* ``sex``: 0 = female, 1 = male (per the upstream README).
* Failed records are appended one-per-line to ``<cache-dir>/error-list.txt``
  (fresh per run); the full failure table lands in ``failures.csv`` at the end.

Usage::

    python scripts/phi_cache_extract.py \
        --data-root /path/to/training_set_small \
        --checkpoint /path/to/SleepPhilosophersStone.ckpt \
        --cache-dir /path/to/phi_cache \
        --workers 4 [--chunked] [--low-freq-cut 1.0]

``--chunked`` switches to the hybrid wavelet stage in ``utils/
phi_preprocess.py`` (low frequencies on the full signal, high frequencies in
overlapping chunks), dropping the peak RAM from ~60-70 GB to ~10 GB so more
record-workers fit in memory.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.phi_cache import PHI_SCORE_KEYS, _resolve_c4m1, cache_path_for_row, record_key

_CHUNKED = False
_CHUNK_SECONDS = 2400.0
_OVERLAP_SECONDS = 600.0
_LOW_FREQ_CUT = 2.0


def _worker_init(
    checkpoint: str,
    device_id: int,
    chunked: bool = False,
    chunk_seconds: float = 2400.0,
    overlap_seconds: float = 600.0,
    low_freq_cut: float = 2.0,
) -> None:
    global _MODEL, _CFG, _CHUNKED, _CHUNK_SECONDS, _OVERLAP_SECONDS, _LOW_FREQ_CUT
    _CHUNKED = chunked
    _CHUNK_SECONDS = chunk_seconds
    _OVERLAP_SECONDS = overlap_seconds
    _LOW_FREQ_CUT = low_freq_cut
    phi_src = PROJECT_ROOT / "third_party" / "philosophers-stone" / "src"
    if str(phi_src) not in sys.path:
        sys.path.insert(0, str(phi_src))
    import torch
    from philosophers_stone.philosopher_utils import Config, load_model

    _CFG = Config(model_file=checkpoint)
    _CFG.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    _MODEL = load_model(_CFG)


def _infer_chunked(signal: np.ndarray, fs: float, age: float, sex: int, file_id: str, collect_heads: bool):
    """Chunked-wavelet path: same pre/post-processing, low-memory CWT."""
    from utils.phi_preprocess import infer_brain_health_chunked

    try:
        return infer_brain_health_chunked(
            signal,
            fs,
            age,
            sex,
            file_id,
            _CFG,
            _MODEL,
            collect_head_outputs=collect_heads,
            chunk_seconds=_CHUNK_SECONDS,
            overlap_seconds=_OVERLAP_SECONDS,
            low_freq_cut=_LOW_FREQ_CUT,
        )
    except Exception as exc:  # noqa: BLE001 — keep the record-level fail semantics
        return {"status": "error", "error_message": str(exc)}


def _process_one(args):
    row, cache_dir, collect_heads, max_seconds = args
    cache_dir = Path(cache_dir)
    out_path = cache_path_for_row(cache_dir, row)
    if out_path.exists():
        return ("skip", record_key(row), None)
    edf_path = Path(row["edf_path"])
    if not edf_path.exists():
        return ("fail", record_key(row), f"missing EDF: {edf_path}")
    resolved = _resolve_c4m1(edf_path)
    if resolved is None:
        return ("fail", record_key(row), "no usable C4-M1 channel")
    signal, fs = resolved
    if not np.isfinite(signal).all():
        return ("fail", record_key(row), "non-finite EEG")
    if max_seconds is not None:
        signal = signal[: int(max_seconds * fs)]

    sex = 1 if str(row.get("Sex")).lower().startswith("male") else 0
    if _CHUNKED:
        result = _infer_chunked(signal, fs, float(row["Age"]), sex, record_key(row), collect_heads)
    else:
        from philosophers_stone.philosopher_utils import infer_brain_health

        result = infer_brain_health(
            signal,
            fs_hz=fs,
            age=float(row["Age"]),
            sex=sex,
            file_id=record_key(row),
            cfg=_CFG,
            model=_MODEL,
            collect_head_outputs=collect_heads,
        )
    if result.get("status") != "ok":
        return ("fail", record_key(row), str(result.get("error_message")))

    save = {
        "latent": np.asarray(result["latent"], dtype=np.float32).reshape(-1),
    }
    for key in PHI_SCORE_KEYS:
        save[key] = float(result[key])
    if collect_heads and isinstance(result.get("predictions"), dict):
        save["heads"] = np.asarray([float(v) for v in result["predictions"].values()], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" to names that do not end in ".npz",
    # so the temporary name must itself end in ".npz".
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **save)
    tmp_path.replace(out_path)
    return ("ok", record_key(row), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="path to SleepPhilosophersStone.ckpt")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="compute the wavelet stage in overlapping chunks (low memory, see utils/phi_preprocess.py)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2400.0, help="chunk length in seconds (chunked mode)")
    parser.add_argument("--overlap-seconds", type=float, default=600.0, help="overlap dropped at each chunk boundary")
    parser.add_argument(
        "--low-freq-cut",
        type=float,
        default=2.0,
        help="frequencies at/below this (Hz) are computed on the full signal",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "truncate each signal to at most this many seconds (smoke/CI use only; "
            "results are NOT comparable to full-night latents)"
        ),
    )
    parser.add_argument("--collect-heads", action="store_true", default=True)
    parser.add_argument("--no-collect-heads", dest="collect_heads", action="store_false")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    error_list = cache_dir / "error-list.txt"
    error_list.write_text("")  # fresh per run; failures are appended as they occur
    demo = pd.read_csv(data_root / "demographics.csv")
    demo = demo.sort_values(["SiteID", "BidsFolder"], kind="stable").reset_index(drop=True)
    demo["edf_path"] = demo.apply(
        lambda r: str(data_root / "physiological_data" / r["SiteID"] / f"{r['BidsFolder']}_ses-{r['SessionID']}.edf"),
        axis=1,
    )
    missing = ~demo["edf_path"].apply(lambda p: Path(p).exists())
    if missing.any():
        print(
            f"dropping {int(missing.sum())} row(s) with missing EDF files " "(partial data-root?)",
            flush=True,
        )
        demo = demo[~missing].reset_index(drop=True)
    if args.limit:
        demo = demo.head(args.limit)
    print(f"records: {len(demo)}, workers: {args.workers}, cache: {cache_dir}", flush=True)

    tasks = [(row, str(cache_dir), args.collect_heads, args.max_seconds) for _, row in demo.iterrows()]
    n_ok = n_skip = 0
    failures = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(
                args.checkpoint,
                args.device_id,
                args.chunked,
                args.chunk_seconds,
                args.overlap_seconds,
                args.low_freq_cut,
            ),
        ) as pool:
            for i, (status, key, err) in enumerate(pool.map(_process_one, tasks)):
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    failures.append((key, err))
                    with open(error_list, "a") as fh:
                        fh.write(f"{key}\n")
                if (i + 1) % 10 == 0:
                    print(f"processed {i + 1}/{len(tasks)} (ok {n_ok}, skip {n_skip})", flush=True)
    else:
        _worker_init(
            args.checkpoint,
            args.device_id,
            args.chunked,
            args.chunk_seconds,
            args.overlap_seconds,
            args.low_freq_cut,
        )
        for i, task in enumerate(tasks):
            status, key, err = _process_one(task)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                failures.append((key, err))
                with open(error_list, "a") as fh:
                    fh.write(f"{key}\n")
            if (i + 1) % 10 == 0:
                print(f"processed {i + 1}/{len(tasks)} (ok {n_ok}, skip {n_skip})", flush=True)

    if failures:
        fail_df = pd.DataFrame(failures, columns=["record", "error"])
        fail_df.to_csv(cache_dir / "failures.csv", index=False)
        print(f"done: ok {n_ok}, skip {n_skip}, failed {len(failures)}", flush=True)
    else:
        print(f"done: ok {n_ok}, skip {n_skip}, failed 0", flush=True)


if __name__ == "__main__":
    main()
