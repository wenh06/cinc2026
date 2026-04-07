"""
Hyperparameter and architecture search script.

Usage (run from repo root, inside a tmux session):
    python utils/run_search.py \
        --data-dir /Data1/wenh06/physionetchallenge2026data \
        --out-dir saved_models/search \
        [--dry-run]   # print experiment list without running

Each experiment trains a model with a specific config and saves results in:
    <out-dir>/<exp_name>/

After all runs:
    python utils/analyze_logs.py --log-dir saved_models/search --no-plot

Supports repeats: each (config, seed) pair is one experiment.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

# Each entry: (field_name, list_of_values)
# These are applied as overrides on top of TrainCfg / ModelCfg.
SEARCH_SPACE = {
    # Model architecture (maps to TrainCfg.model_name)
    "model_name": [
        "epoch_crnn_S",
        "epoch_crnn_M",
        "epoch_crnn_L",
        "epoch_transformer_S",
        "epoch_transformer_M",
        "epoch_transformer_L",
        "epoch_crnn_resnetNC_BNse_S",
        "epoch_crnn_resnetNC_BNse_M",
        # tresnet variants — heavier, include only if time allows
        # "epoch_crnn_tresnetE_S",
        # "epoch_crnn_tresnetE_M",
    ],
    # Label smoothing
    "label_smoothing": [0.0, 0.05, 0.10],
    # OneCycleLR warm-up fraction
    "pct_start": [0.1, 0.3],
    # AdamW weight decay
    "weight_decay": [1e-3, 1e-2],
}

# Random seeds for repeat experiments
SEEDS = [42, 123, 0]

# Fixed training settings (override TrainCfg defaults)
FIXED = {
    "n_epochs": 100,
    "batch_size": 16,
    "lr": 3e-4,
    "max_lr": 1e-3,
    "lr_scheduler": "one_cycle",
    "grad_clip": 1.0,
    "early_stopping_patience": 20,
}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def build_experiments(search_space: dict, seeds: list) -> list:
    """Generate all (config, seed) combinations."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    experiments = []
    for combo in product(*values):
        cfg = dict(zip(keys, combo))
        for seed in seeds:
            experiments.append({**cfg, "seed": seed})
    return experiments


def exp_name(exp: dict) -> str:
    """Build a short, filesystem-safe name for an experiment."""
    parts = [
        exp["model_name"],
        f"ls{exp['label_smoothing']}",
        f"pct{exp['pct_start']}",
        f"wd{exp['weight_decay']}",
        f"seed{exp['seed']}",
    ]
    return "_".join(str(p) for p in parts)


def already_done(out_path: Path) -> bool:
    """Return True if this experiment has already produced a final model."""
    return (out_path / "final_model.pth.tar").exists()


def run_experiment(exp: dict, data_dir: str, out_dir: str, dry_run: bool = False) -> bool:
    """Train one experiment.  Returns True on success."""
    name = exp_name(exp)
    model_folder = str(Path(out_dir) / name)

    if already_done(Path(model_folder)):
        print(f"[SKIP] {name} — already finished")
        return True

    # Write a per-experiment cfg override file that train_model.py will read
    override = {**FIXED, **{k: v for k, v in exp.items()}}
    override["db_dir"] = data_dir
    override["model_folder"] = model_folder

    os.makedirs(model_folder, exist_ok=True)
    override_path = Path(model_folder) / "override.json"
    with open(override_path, "w") as f:
        json.dump(override, f, indent=2)

    env = {**os.environ}
    # Pass overrides as environment variables
    env["CINC2026_OVERRIDE_JSON"] = str(override_path)

    cmd = [
        sys.executable,
        "train_model.py",
        "-d",
        data_dir,
        "-m",
        model_folder,
        "-v",
    ]

    print(f"\n{'='*70}")
    print(f"[START] {name}")
    print(f"  cmd : {' '.join(cmd)}")
    print(f"  cfg : {json.dumps({k: v for k, v in exp.items()}, separators=(',', ':'))}")
    print(f"  time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if dry_run:
        print("[DRY-RUN] skipping actual training")
        return True

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent),  # repo root
            timeout=3 * 3600,  # 3-hour cap per experiment
        )
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"[{status}] {name} — {elapsed/60:.1f} min")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} exceeded 3 h — skipping")
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter / architecture search")
    parser.add_argument("--data-dir", required=True, help="Path to PhysioNet 2026 training_set")
    parser.add_argument("--out-dir", default="saved_models/search", help="Root output folder")
    parser.add_argument("--dry-run", action="store_true", help="Print experiments without running")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds for repeats")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict model_name to these values (subset of SEARCH_SPACE['model_name'])",
    )
    parser.add_argument(
        "--label-smoothing",
        nargs="+",
        type=float,
        default=None,
        help="Restrict label_smoothing values",
    )
    parser.add_argument(
        "--skip-repeats",
        action="store_true",
        help="Use only the first seed (no repeats), useful for a quick first pass",
    )
    args = parser.parse_args()

    space = deepcopy(SEARCH_SPACE)
    if args.models:
        space["model_name"] = [m for m in args.models if m in SEARCH_SPACE["model_name"]]
    if args.label_smoothing:
        space["label_smoothing"] = args.label_smoothing

    seeds = [args.seeds[0]] if args.skip_repeats else args.seeds

    experiments = build_experiments(space, seeds)

    print(f"Total experiments: {len(experiments)}")
    print(f"Seeds: {seeds}")
    if args.dry_run:
        print("\n--- Experiment list ---")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i:3d}. {exp_name(exp)}")
        return

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Write a manifest
    manifest_path = Path(args.out_dir) / "manifest.json"
    manifest = [{"name": exp_name(e), **e} for e in experiments]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    results = {}
    for i, exp in enumerate(experiments, 1):
        name = exp_name(exp)
        print(f"\n[{i}/{len(experiments)}] {name}")
        ok = run_experiment(exp, args.data_dir, args.out_dir, dry_run=args.dry_run)
        results[name] = "ok" if ok else "failed"

    # Summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    ok_count = sum(1 for v in results.values() if v == "ok")
    print(f"  {ok_count}/{len(results)} experiments succeeded")
    failed = [k for k, v in results.items() if v != "ok"]
    if failed:
        print(f"  Failed: {failed}")

    summary_path = Path(args.out_dir) / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
