"""Training log analysis utilities for CinC 2026.

Each training run produces a CSV file under ``working_dir/log/`` with columns::

    step, time, part, epoch, loss, lr, auroc, auprc,
    auroc_I0002, auroc_I0006, auroc_S0001

* ``part == "train"``  rows carry ``loss`` and ``lr`` (metrics are empty).
* ``part == "val"``    rows carry ``auroc``, ``auprc``, ``auroc_*``  (loss is empty).

Usage
-----
From the command line::

    # summary table + plots for all runs in a log directory
    python utils/analyze_logs.py --log-dir saved_models/run/working_dir/log

    # compare two specific runs, save plots to results/
    python utils/analyze_logs.py \\
        --log-dir saved_models/run/working_dir/log \\
        --out-dir results/plots

    # print best-checkpoint table only (no plots)
    python utils/analyze_logs.py --log-dir ... --no-plot

From a notebook::

    from utils.analyze_logs import load_runs, plot_learning_curves, best_checkpoints
    runs = load_runs("saved_models/run/working_dir/log")
    plot_learning_curves(runs)
    print(best_checkpoints(runs))
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ── Site columns present in every log ──────────────────────────────────────
SITE_COLS = ["auroc_I0002", "auroc_I0006", "auroc_S0001"]
SITE_LABELS = {"auroc_I0002": "I0002 (in-lab)", "auroc_I0006": "I0006 (in-lab)", "auroc_S0001": "S0001 (home)"}

# Regex to extract a short human-readable run name from the filename
_RUN_NAME_RE = re.compile(r"TorchECG_(\d{2}-\d{2}_\d{2}-\d{2})_(.*?)\.csv$")


# ── Public API ──────────────────────────────────────────────────────────────


def load_runs(log_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load all ``*.csv`` training logs in *log_dir*.

    Returns
    -------
    dict
        ``{run_name: dataframe}`` where *dataframe* contains the raw rows
        (both ``"train"`` and ``"val"`` parts).  Columns are typed correctly
        and empty cells become ``NaN``.
    """
    log_dir = Path(log_dir)
    csvs = sorted(log_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")

    runs: Dict[str, pd.DataFrame] = {}
    for path in csvs:
        m = _RUN_NAME_RE.search(path.name)
        run_name = m.group(1) if m else path.stem
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        # coerce numeric columns (empty strings → NaN)
        for col in ["loss", "lr", "auroc", "auprc"] + SITE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        runs[run_name] = df
    return runs


def val_df(run: pd.DataFrame) -> pd.DataFrame:
    """Return per-epoch validation rows, one row per epoch."""
    v = run[run["part"] == "val"].copy()
    # Keep only the last val entry per epoch (in case of restarts mid-epoch)
    v = v.sort_values("step").drop_duplicates(subset="epoch", keep="last")
    return v.reset_index(drop=True)


def train_df(run: pd.DataFrame) -> pd.DataFrame:
    """Return training rows (multiple steps per epoch, one row per log_step)."""
    return run[run["part"] == "train"].copy().reset_index(drop=True)


def epoch_train_loss(run: pd.DataFrame) -> pd.DataFrame:
    """Mean training loss per epoch."""
    t = train_df(run)
    return t.groupby("epoch")["loss"].mean().reset_index()


def best_checkpoints(runs: Dict[str, pd.DataFrame], metric: str = "auroc") -> pd.DataFrame:
    """Return the best-epoch summary for every run.

    Parameters
    ----------
    metric
        Column to maximise (default ``"auroc"``).

    Returns
    -------
    DataFrame with columns: run, best_epoch, auroc, auprc, auroc_I0002,
    auroc_I0006, auroc_S0001, total_epochs.
    """
    rows = []
    for name, run in runs.items():
        v = val_df(run)
        if v.empty or v[metric].isna().all():
            continue
        idx = v[metric].idxmax()
        best = v.loc[idx]
        rows.append(
            {
                "run": name,
                "best_epoch": int(best["epoch"]),  # type: ignore
                "auroc": round(best["auroc"], 4),
                "auprc": round(best["auprc"], 4),
                "auroc_I0002": round(best["auroc_I0002"], 4),
                "auroc_I0006": round(best["auroc_I0006"], 4),
                "auroc_S0001": round(best["auroc_S0001"], 4),
                "total_epochs": int(v["epoch"].max()) + 1,
            }
        )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_learning_curves(
    runs: Dict[str, pd.DataFrame],
    smooth: int = 3,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:  # type: ignore
    """Plot train-loss and val-AUROC/AUPRC learning curves for all runs.

    Parameters
    ----------
    smooth
        Rolling-window half-width for smoothing.  ``0`` = no smoothing.
    out_path
        If given, save figure here (PDF).
    """
    n_runs = len(runs)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_runs))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_loss, ax_auroc, ax_auprc = axes

    for (name, run), color in zip(runs.items(), colors):
        v = val_df(run)
        tl = epoch_train_loss(run)

        if smooth > 0:
            tl_plot = _smooth(tl["loss"], smooth)
            v_auroc = _smooth(v["auroc"], smooth)
            v_auprc = _smooth(v["auprc"], smooth)
        else:
            tl_plot = tl["loss"]
            v_auroc = v["auroc"]
            v_auprc = v["auprc"]

        kw = dict(color=color, linewidth=1.5, label=name)
        ax_loss.plot(tl["epoch"], tl_plot, **kw)
        ax_auroc.plot(v["epoch"], v_auroc, **kw)
        ax_auprc.plot(v["epoch"], v_auprc, **kw)

        # Mark best epoch
        if not v["auroc"].isna().all():
            best_i = v["auroc"].idxmax()
            ax_auroc.axvline(v.loc[best_i, "epoch"], color=color, linestyle=":", alpha=0.7)

    ax_loss.set(title="Train loss (per-epoch mean)", xlabel="Epoch", ylabel="BCE loss")
    ax_auroc.set(title="Val AUROC", xlabel="Epoch", ylabel="AUROC")
    ax_auprc.set(title="Val AUPRC", xlabel="Epoch", ylabel="AUPRC")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Learning curves", fontsize=13, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


def plot_site_auroc(
    runs: Dict[str, pd.DataFrame],
    smooth: int = 3,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:  # type: ignore
    """Per-site AUROC curves — useful for diagnosing site-specific generalisation."""
    n_runs = len(runs)
    n_sites = len(SITE_COLS)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_runs))
    fig, axes = plt.subplots(1, n_sites, figsize=(5 * n_sites, 4), sharey=True)

    for ax, site_col in zip(axes, SITE_COLS):
        for (name, run), color in zip(runs.items(), colors):
            v = val_df(run)
            if v[site_col].isna().all():
                continue
            series = _smooth(v[site_col], smooth) if smooth > 0 else v[site_col]
            ax.plot(v["epoch"], series, color=color, linewidth=1.5, label=name)
        ax.set(title=SITE_LABELS.get(site_col, site_col), xlabel="Epoch", ylabel="AUROC")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-site validation AUROC", fontsize=13, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


def plot_lr_schedule(
    runs: Dict[str, pd.DataFrame],
    out_path: Optional[str | Path] = None,
) -> plt.Figure:  # type: ignore
    """LR schedule for each run."""
    n_runs = len(runs)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_runs))
    fig, ax = plt.subplots(figsize=(8, 3))
    for (name, run), color in zip(runs.items(), colors):
        t = train_df(run)
        ax.plot(t["step"], t["lr"], color=color, linewidth=1.2, label=name)
    ax.set(title="Learning-rate schedule", xlabel="Global step", ylabel="LR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


def print_run_summary(runs: Dict[str, pd.DataFrame]) -> None:
    """Print a concise text summary of all runs."""
    print("\n" + "=" * 70)
    print("Best-checkpoint summary (sorted by val AUROC)")
    print("=" * 70)
    bc = best_checkpoints(runs)
    print(bc.to_string(index=False))
    print()

    for name, run in runs.items():
        v = val_df(run)
        t = train_df(run)
        n_epochs = int(v["epoch"].max()) + 1 if not v.empty else 0
        duration = (run["time"].max() - run["time"].min()).total_seconds() / 60
        print(f"  {name}")
        print(f"    epochs={n_epochs}  train_steps={len(t)}  duration≈{duration:.1f} min")
        print(f"    final val AUROC={v['auroc'].iloc[-1]:.4f}  AUPRC={v['auprc'].iloc[-1]:.4f}")
        if not v["auroc"].isna().all():
            best_i = v["auroc"].idxmax()
            print(
                f"    best  val AUROC={v.loc[best_i,'auroc']:.4f} @ epoch {int(v.loc[best_i,'epoch'])}"  # type: ignore
                f"  (AUPRC={v.loc[best_i,'auprc']:.4f})"
            )
        print()


def compare_metrics_table(runs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-epoch val metrics for all runs merged into a single tidy DataFrame.

    Useful for detailed analysis in a notebook.
    """
    frames: List[pd.DataFrame] = []
    for name, run in runs.items():
        v = val_df(run).copy()
        v.insert(0, "run", name)
        frames.append(v)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default="saved_models/run/working_dir/log", help="Directory containing *.csv log files")
    p.add_argument("--out-dir", default=None, help="Directory to save plots (default: show interactively)")
    p.add_argument("--smooth", type=int, default=3, help="Rolling-average window for curves (0=off)")
    p.add_argument("--no-plot", action="store_true", help="Print summary table only, skip plotting")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    runs = load_runs(args.log_dir)
    print_run_summary(runs)

    if args.no_plot:
        return

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _savepath(name: str) -> Optional[Path]:
        return out_dir / name if out_dir else None

    plot_learning_curves(runs, smooth=args.smooth, out_path=_savepath("learning_curves.pdf"))
    plot_site_auroc(runs, smooth=args.smooth, out_path=_savepath("site_auroc.pdf"))
    plot_lr_schedule(runs, out_path=_savepath("lr_schedule.pdf"))

    if not out_dir:
        plt.show()


if __name__ == "__main__":
    main()
