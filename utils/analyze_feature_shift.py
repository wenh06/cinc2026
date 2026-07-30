"""
analyze_feature_shift.py
========================
Analyse CAISR epoch feature distributions across training sites to estimate
how much covariate shift to expect on hidden validation/test sets.

Analysis 1 (main): Inter-site CAISR feature comparison
  S0001 vs I0002 vs I0006  (all three training sites)

Analysis 2: Supplementary-set RAW signal statistics vs training sites
  (no CAISR annotations in supplementary_set; compare raw EDF header stats)

Usage::

    python utils/analyze_feature_shift.py \\
        --data-dir /Data1/wenh06/physionetchallenge2026data
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data_reader import CINC2026
from dataset import build_epoch_features

FEATURE_NAMES = [
    "stage_N3",
    "stage_N2",
    "stage_N1",
    "stage_REM",
    "stage_W",
    "stage_Unk",
    "prob_N3",
    "prob_N2",
    "prob_N1",
    "prob_REM",
    "prob_W",
    "arousal_frac",
    "resp_OA",
    "resp_CA",
    "resp_MA",
    "resp_HY",
    "resp_RERA",
    "limb_iso",
    "limb_PLM",
    "pos_sin",
    "pos_cos",
]
assert len(FEATURE_NAMES) == 21
FEAT_ANALYSIS = FEATURE_NAMES[:19]  # exclude time-encoding


# ── helpers ────────────────────────────────────────────────────────────────────


def record_mean_features(dr: CINC2026, site: str) -> pd.DataFrame:
    """Per-record mean of each CAISR feature for a given site."""
    rows = []
    df = dr._df_records.reset_index()
    df = df[df["SiteID"] == site]
    print(f"  Site {site}: {len(df)} records", flush=True)

    for row in df.itertuples():
        rec = row.BidsFolder
        ann = dr.load_ann(rec, ann_type="algorithmic")
        if not ann or "stage_caisr" not in ann:
            continue
        feats = build_epoch_features(ann)
        if len(feats) == 0:
            continue
        means = feats[:, :19].mean(axis=0)
        d = {"record_id": rec, "site": site, "n_epochs": len(feats)}
        for j, name in enumerate(FEAT_ANALYSIS):
            d[name] = float(means[j])
        rows.append(d)

    return pd.DataFrame(rows)


def epoch_features_for_site(dr: CINC2026, site: str) -> np.ndarray:
    """Stack all epoch features for a site → shape (N_total_epochs, 21)."""
    chunks = []
    df = dr._df_records.reset_index()
    df = df[df["SiteID"] == site]
    for row in df.itertuples():
        ann = dr.load_ann(row.BidsFolder, ann_type="algorithmic")
        if not ann or "stage_caisr" not in ann:
            continue
        feats = build_epoch_features(ann)
        if len(feats):
            chunks.append(feats)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 21), np.float32)


def ks_pairwise(arr_a: np.ndarray, arr_b: np.ndarray, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(FEAT_ANALYSIS):
        stat, pval = ks_2samp(arr_a[:, j], arr_b[:, j])
        rows.append(
            {
                "feature": name,
                "comparison": f"{label_a} vs {label_b}",
                "ks_stat": round(stat, 4),
                "p_value": round(pval, 5),
                "sig": pval < 0.05,
            }
        )
    return pd.DataFrame(rows)


def raw_signal_stats(edf_path: Path) -> dict | None:
    """Read first 5 channels from an EDF and return their physical range / fs."""
    try:
        from pyedflib import EdfReader

        r = EdfReader(str(edf_path))
        labels = r.getSignalLabels()
        stats = {"n_channels": r.signals_in_file, "duration_s": r.getFileDuration()}
        fs_list = []
        for i in range(min(5, r.signals_in_file)):
            fs_list.append(r.getSampleFrequency(i))
        stats["fs_sample"] = fs_list
        stats["channel_sample"] = labels[:5]
        r._close()
        return stats
    except Exception:
        return None


# ── plots ───────────────────────────────────────────────────────────────────────


def plot_boxplots(dfs: dict[str, pd.DataFrame], out_path: Path):
    sites = list(dfs.keys())
    n_feat = len(FEAT_ANALYSIS)
    n_cols = 4
    n_rows = (n_feat + n_cols - 1) // n_cols
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for j, name in enumerate(FEAT_ANALYSIS):
        ax = axes[j]
        data = [dfs[s][name].dropna().values for s in sites]
        bp = ax.boxplot(data, labels=sites, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CAISR Feature Distributions by Training Site\n(per-record means)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_kde_grid(epoch_arrays: dict[str, np.ndarray], out_path: Path):
    from scipy.stats import gaussian_kde

    colors = {"S0001": "#4C72B0", "I0002": "#DD8452", "I0006": "#55A868"}
    selected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 17]
    selected_names = [FEATURE_NAMES[j] for j in selected]

    n_cols = 4
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for ax_i, (feat_j, name) in enumerate(zip(selected, selected_names)):
        ax = axes[ax_i]
        for site, arr in epoch_arrays.items():
            col = arr[:, feat_j]
            col = col[~np.isnan(col)]
            if len(col) < 10 or col.std() < 1e-9:
                continue
            x = np.linspace(col.min(), col.max(), 200)
            try:
                kde = gaussian_kde(col)
                ax.plot(x, kde(x), label=site, color=colors.get(site, "gray"), linewidth=1.5)
            except Exception:
                pass
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Epoch-Level KDE by Training Site", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── supplementary set raw signal analysis ──────────────────────────────────────


def analyze_supplementary_raw(data_dir: Path, out_dir: Path):
    """Compare raw EDF channel availability/fs across training and supplementary sites."""
    rows = []

    def scan_site(site_dir: Path, partition: str):
        for edf in sorted(site_dir.glob("*.edf"))[:5]:  # sample 5 per site
            stats = raw_signal_stats(edf)
            if stats:
                rows.append(
                    {
                        "partition": partition,
                        "site": site_dir.name,
                        "file": edf.name,
                        "n_channels": stats["n_channels"],
                        "duration_s": stats["duration_s"],
                        "fs_sample": str(stats["fs_sample"]),
                        "channels_sample": str(stats["channel_sample"]),
                    }
                )

    # Auto-detect training partition (official phase: training_set_small/large)
    physio_train = None
    for part in ["training_set_small", "training_set_large", "training_set"]:
        candidate = data_dir / part / "physiological_data"
        if candidate.exists():
            physio_train = candidate
            break
    if physio_train is None:
        print("No training partition found; skipping physiological data scan.", file=sys.stderr)
        physio_train = Path("/nonexistent")  # empty iterator below
    physio_supp = data_dir / "supplementary_set" / "physiological_data"

    for site_dir in sorted(physio_train.iterdir()):
        if site_dir.is_dir():
            scan_site(site_dir, "training_set")

    for site_dir in sorted(physio_supp.iterdir()):
        if site_dir.is_dir():
            scan_site(site_dir, "supplementary_set")

    df = pd.DataFrame(rows)
    print("\n--- Raw EDF signal inventory (sample) ---")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "raw_signal_inventory.csv", index=False)
    return df


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Data1/wenh06/physionetchallenge2026data")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "utils"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data reader ...")
    dr = CINC2026(db_dir=data_dir, verbose=0)
    sites_train = ["S0001", "I0002", "I0006"]

    # ── Analysis 1: CAISR feature shift across training sites ─────────────────
    print("\n=== Analysis 1: CAISR feature distributions by training site ===")
    dfs_record = {}
    epoch_arrays = {}
    for site in sites_train:
        print(f"\nSite {site} — per-record means:")
        dfs_record[site] = record_mean_features(dr, site)
        print(f"  Site {site} — loading epoch-level arrays ...")
        epoch_arrays[site] = epoch_features_for_site(dr, site)
        print(f"  {site}: {epoch_arrays[site].shape[0]:,} epochs")

    # Mean comparison table
    rows = []
    for name in FEAT_ANALYSIS:
        row = {"feature": name}
        for site in sites_train:
            vals = dfs_record[site][name].dropna()
            row[f"{site}_mean"] = round(vals.mean(), 4)
            row[f"{site}_std"] = round(vals.std(), 4)
        rows.append(row)
    df_means = pd.DataFrame(rows)

    # Add max range across sites as a shift proxy
    df_means["max_site_delta"] = df_means[[f"{s}_mean" for s in sites_train]].max(axis=1) - df_means[
        [f"{s}_mean" for s in sites_train]
    ].min(axis=1)
    df_means = df_means.sort_values("max_site_delta", ascending=False)
    print("\n--- Per-feature site-mean comparison (sorted by max_site_delta) ---")
    print(df_means.to_string(index=False))
    df_means.to_csv(out_dir / "feature_site_means.csv", index=False)

    # KS tests between each pair
    ks_rows = []
    pairs = [("S0001", "I0002"), ("S0001", "I0006"), ("I0002", "I0006")]
    for a, b in pairs:
        ks = ks_pairwise(epoch_arrays[a], epoch_arrays[b], a, b)
        ks_rows.append(ks)
    df_ks = pd.concat(ks_rows, ignore_index=True)
    df_ks_wide = df_ks.pivot(index="feature", columns="comparison", values="ks_stat").reset_index()
    df_ks_wide["max_ks"] = df_ks_wide[[f"{a} vs {b}" for a, b in pairs]].max(axis=1)
    df_ks_wide = df_ks_wide.sort_values("max_ks", ascending=False)
    print("\n--- KS statistics between training sites (epoch-level) ---")
    print(df_ks_wide.to_string(index=False))
    df_ks_wide.to_csv(out_dir / "ks_between_sites.csv", index=False)

    # Plots
    print("\nGenerating plots ...")
    plot_boxplots(dfs_record, out_dir / "feature_shift_boxplot.png")
    plot_kde_grid(epoch_arrays, out_dir / "feature_shift_kde.png")

    # ── Analysis 2: Raw signal inventory (supplementary set) ──────────────────
    print("\n=== Analysis 2: Raw EDF inventory (supplementary set vs training) ===")
    analyze_supplementary_raw(data_dir, out_dir)

    # ── Summary: which features are most affected by site shift ──────────────
    print("\n=== Summary: Top features with highest cross-site shift ===")
    top = df_ks_wide.nlargest(10, "max_ks")[["feature", "max_ks"]]
    print(top.to_string(index=False))
    print("\nThese features are most likely to cause domain shift at I0004/I0007.")
    print("Consider site-specific normalization or dropping these as input features.")

    print("\nDone. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
