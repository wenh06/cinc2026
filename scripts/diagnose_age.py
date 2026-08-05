"""
Age dependence diagnostics for CinC 2026 baseline.

Answers the question: is the model using age as a shortcut?

Generates three diagnostic plots:
  1. Predicted probability vs age (scatter + LOESS), colored by true CI label.
  2. Mean predicted prob by age bin, with 95% CI.
  3. AUROC by age bin — does model performance degrade at certain ages?
  4. Per-site age-prob relationship.

Usage:
    python tools/diagnose_age.py \
        -d data/official-phase-large \
        -m saved_models/official_baseline/BestModel_EpochCRNN-epoch51_....pth.tar \
        -o images/age_diagnosis/
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from cfg import TrainCfg
from dataset import CINC2026Dataset, collate_fn
from models import EpochCRNN  # noqa: E402
from utils.misc import age_conditioned_auroc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Age dependence diagnostics for CinC 2026 baseline")
    parser.add_argument("-d", "--data-dir", type=str, default="data/official-phase-large", dest="data_dir")
    parser.add_argument("-s", "--split-file", type=str, default="utils/cinc2026-data-split.json", dest="split_file")
    parser.add_argument("-m", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-o", "--output-dir", type=str, default="images/age_diagnosis", dest="output_dir")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--age-bin-width", type=float, default=5.0, dest="age_bin_width")
    return parser.parse_args()


def load_model(model_path: str, device: str):
    """Load the best model checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_config = deepcopy(ckpt["model_config"])
    model = EpochCRNN(**model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device) -> pd.DataFrame:
    """Run inference on the validation set, returning a DataFrame with predictions."""
    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            output = model(batch)
            probs = output["ci_prob"][:, 1].cpu().numpy()  # probability of positive class
            labels = batch["label"].cpu().numpy()
            ages = batch["demographics"][:, 0].cpu().numpy() * 100.0  # denormalize: age/100 → age
            sites = batch["site_id"]  # list of str
            for i in range(len(probs)):
                records.append(
                    {
                        "prob": float(probs[i]),
                        "label": int(labels[i]),
                        "age": float(ages[i]),
                        "site": str(sites[i]),
                    }
                )
    return pd.DataFrame(records)


def plot_prob_vs_age(df: pd.DataFrame, output_path: str):
    """Plot A: Predicted probability vs age, colored by true CI label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter with LOESS
    ax = axes[0]
    for label, color, marker, name in [(0, "#2166ac", "o", "CI Negative"), (1, "#b2182b", "x", "CI Positive")]:
        subset = df[df["label"] == label]
        ax.scatter(subset["age"], subset["prob"], c=color, marker=marker, alpha=0.4, s=12, label=name, edgecolors="none")

    # LOESS smooth for each class
    ages_sorted = np.linspace(df["age"].min(), df["age"].max(), 100)
    for label, color, ls in [(0, "#2166ac", "--"), (1, "#b2182b", "-")]:
        subset = df[df["label"] == label]
        if len(subset) > 5:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(subset["age"], subset["prob"])
            ax.plot(ages_sorted, iso.predict(ages_sorted), c=color, ls=ls, lw=2, alpha=0.9)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Predicted CI Probability")
    # ax.set_title("Predicted Probability vs Age (by true label)")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.02, 1.02)

    # Right: all subjects combined, LOESS
    ax = axes[1]
    ax.scatter(
        df["age"],
        df["prob"],
        c=["#2166ac" if lab == 0 else "#b2182b" for lab in df["label"]],
        alpha=0.3,
        s=10,
        edgecolors="none",
    )
    iso_all = IsotonicRegression(out_of_bounds="clip")
    iso_all.fit(df["age"], df["prob"])
    ax.plot(ages_sorted, iso_all.predict(ages_sorted), "k-", lw=2, label="LOESS (all)")
    # Horizontal line at prevalence
    prevalence = df["label"].mean()
    ax.axhline(prevalence, color="gray", ls=":", lw=1, label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Predicted CI Probability")
    # ax.set_title("Predicted Probability vs Age (all subjects)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_mean_prob_by_age(df: pd.DataFrame, bin_width: float, output_path: str):
    """Plot B: Mean predicted probability by age bins, with 95% CI."""
    age_min, age_max = np.floor(df["age"].min()), np.ceil(df["age"].max())
    bins = np.arange(age_min, age_max + bin_width, bin_width)
    df["age_bin"] = pd.cut(df["age"], bins=bins, right=False)
    bin_centers = bins[:-1] + bin_width / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean prob by age bin
    ax = axes[0]
    stats = df.groupby("age_bin", observed=False).agg(
        mean_prob=("prob", "mean"),
        sem=("prob", "sem"),
        count=("prob", "count"),
        prevalence=("label", "mean"),
    )
    x = bin_centers[: len(stats)]
    ax.fill_between(
        x, stats["mean_prob"] - 1.96 * stats["sem"], stats["mean_prob"] + 1.96 * stats["sem"], alpha=0.2, color="steelblue"
    )
    ax.plot(x, stats["mean_prob"], "o-", c="steelblue", lw=2, markersize=6)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean Predicted Probability")
    # ax.set_title("Mean Predicted Probability by Age Bin (95% CI)")

    # Right: count histogram in background, prevalence line
    ax = axes[1]
    ax2 = ax.twinx()
    bars = ax.bar(x, stats["count"], width=bin_width * 0.8, alpha=0.3, color="gray", edgecolor="none")
    ax2.plot(x, stats["prevalence"], "s-", c="#b2182b", lw=2, markersize=6, label="True prevalence")
    ax2.plot(x, stats["mean_prob"], "o-", c="steelblue", lw=2, markersize=6, label="Mean pred prob")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("# Subjects", color="gray")
    ax2.set_ylabel("Probability / Prevalence")
    # ax.set_title("Subject Count, Prevalence, and Prediction by Age")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_auroc_by_age(df: pd.DataFrame, bin_width: float, output_path: str):
    """Plot C: AUROC and age-conditioned AUROC by age bin."""
    age_min, age_max = np.floor(df["age"].min()), np.ceil(df["age"].max())
    bins = np.arange(age_min, age_max + bin_width, bin_width)
    df["age_bin"] = pd.cut(df["age"], bins=bins, right=False)
    bin_centers = bins[:-1] + bin_width / 2

    records = []
    for i, (bin_label, group) in enumerate(df.groupby("age_bin", observed=False)):
        if len(group) < 5 or group["label"].nunique() < 2:
            records.append({"age_center": bin_centers[i], "auroc": np.nan, "n": len(group), "n_pos": int(group["label"].sum())})
            continue
        auroc = roc_auc_score(group["label"], group["prob"])
        records.append({"age_center": bin_centers[i], "auroc": auroc, "n": len(group), "n_pos": int(group["label"].sum())})
    bin_df = pd.DataFrame(records)

    # Age-conditioned AUROC globally
    age_cond = age_conditioned_auroc(df["prob"].values, df["label"].values, df["age"].values, age_tolerance=2.0)
    full_auroc = roc_auc_score(df["label"], df["prob"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_df["age_center"], bin_df["auroc"], "o-", c="steelblue", lw=2, markersize=8, label="AUROC per age bin")
    ax.axhline(full_auroc, color="steelblue", ls="--", lw=1, label=f"Full AUROC = {full_auroc:.4f}")
    ax.axhline(age_cond, color="#b2182b", ls="--", lw=1, label=f"Age-cond AUROC = {age_cond:.4f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chance")

    # Annotate sample sizes
    for _, row in bin_df.iterrows():
        if not np.isnan(row["auroc"]):
            ax.annotate(
                f'n={int(row["n"])}',
                (row["age_center"], row["auroc"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color="gray",
            )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("AUROC")
    # ax.set_title("AUROC by Age Bin")
    ax.legend(loc="lower left")
    ax.set_ylim(0.2, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_site(df: pd.DataFrame, output_path: str):
    """Plot D: Per-site age-prob relationship."""
    sites = sorted(df["site"].unique())
    n_sites = len(sites)
    fig, axes = plt.subplots(1, n_sites, figsize=(5 * n_sites, 4.5), squeeze=False)

    for i, site in enumerate(sites):
        ax = axes[0, i]
        site_df = df[df["site"] == site]
        for label, color, name in [(0, "#2166ac", "CI-"), (1, "#b2182b", "CI+")]:
            subset = site_df[site_df["label"] == label]
            ax.scatter(subset["age"], subset["prob"], c=color, alpha=0.4, s=10, label=name, edgecolors="none")
        # LOESS
        ages_sorted = np.linspace(site_df["age"].min(), site_df["age"].max(), 50)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(site_df["age"], site_df["prob"])
        ax.plot(ages_sorted, iso.predict(ages_sorted), "k-", lw=2)
        ax.set_xlabel("Age")
        ax.set_ylabel("Predicted Prob")
        n_pos = int(site_df["label"].sum())
        n_tot = len(site_df)
        # ax.set_title(f"{site} (n={n_tot}, CI%={n_pos/n_tot*100:.1f}%)")

    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def print_summary(df: pd.DataFrame):
    """Print quantitative diagnostics."""
    print("\n" + "=" * 60)
    print("AGE DEPENDENCE DIAGNOSTICS SUMMARY")
    print("=" * 60)

    full_auroc = roc_auc_score(df["label"], df["prob"])
    age_cond = age_conditioned_auroc(df["prob"].values, df["label"].values, df["age"].values, age_tolerance=2.0)

    print(f"\n  N val subjects: {len(df)}")
    print(f"  CI prevalence:  {df['label'].mean():.4f}")
    print(f"  Mean age:        {df['age'].mean():.1f} ± {df['age'].std():.1f}")
    print()
    print(f"  Full AUROC:      {full_auroc:.4f}")
    print(f"  Age-cond AUROC:  {age_cond:.4f}")
    print(f"  Age gap:         {full_auroc - age_cond:.4f}")
    print()

    # Correlation: age vs prob
    age_prob_corr = np.corrcoef(df["age"], df["prob"])[0, 1]
    print(f"  Pearson r(age, prob):  {age_prob_corr:+.4f}")
    print()

    # By true label: mean prob in different age ranges
    for name, group in [("CI Negative (label=0)", df[df["label"] == 0]), ("CI Positive (label=1)", df[df["label"] == 1])]:
        age_lo = group[group["age"] < group["age"].median()]
        age_hi = group[group["age"] >= group["age"].median()]
        print(f"  {name}:")
        print(f"    N={len(group)}, mean prob={group['prob'].mean():.4f}")
        print(f"    Younger half (age<{group['age'].median():.1f}): mean prob={age_lo['prob'].mean():.4f}")
        print(f"    Older half   (age>={group['age'].median():.1f}): mean prob={age_hi['prob'].mean():.4f}")
        print(f"    Δ prob (old − young): {age_hi['prob'].mean() - age_lo['prob'].mean():+.4f}")

    # Per-site breakdown
    print()
    for site in sorted(df["site"].unique()):
        site_df = df[df["site"] == site]
        if len(site_df) < 2 or site_df["label"].nunique() < 2:
            continue
        s_auroc = roc_auc_score(site_df["label"], site_df["prob"])
        s_age_cond = age_conditioned_auroc(
            site_df["prob"].values, site_df["label"].values, site_df["age"].values, age_tolerance=2.0
        )
        s_corr = np.corrcoef(site_df["age"], site_df["prob"])[0, 1]
        print(
            f"  {site}: AUROC={s_auroc:.4f}, age_AUROC={s_age_cond:.4f}, gap={s_auroc-s_age_cond:.4f}, r(age,prob)={s_corr:+.4f}"
        )

    # Interpretation guide
    print()
    print("  INTERPRETATION:")
    gap = full_auroc - age_cond
    if gap < 0.03:
        print("  ✓ Age gap < 0.03: Model does NOT rely on age as a shortcut.")
    elif gap < 0.07:
        print("  ~ Age gap 0.03-0.07: Mild age dependence. Monitor but not critical.")
    else:
        print("  ⚠ Age gap > 0.07: Significant age dependence. Consider age-adversarial head or age-stratified sampling.")

    if age_prob_corr > 0.3:
        print("  ⚠ High corr(age, prob): Model predictions are strongly age-correlated. Age-adversarial training recommended.")
    elif age_prob_corr > 0.15:
        print("  ~ Moderate corr(age, prob): Some age leakage, may benefit from regularization.")
    else:
        print("  ✓ Low corr(age, prob): Age is not dominating predictions.")

    print("=" * 60)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_file = Path(args.split_file)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir}")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Create dataset & dataloader (val-only)
    val_config = deepcopy(TrainCfg)
    val_config.db_dir = str(data_dir)
    val_config.lazy = False

    val_dataset = CINC2026Dataset(val_config, training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )
    print(f"Val subjects: {len(val_dataset)}")
    print(f"Val batches: {len(val_loader)}")

    # Load model
    model = load_model(str(model_path), args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Run inference
    df = run_inference(model, val_loader, args.device)
    print(f"Inference done: {len(df)} predictions collected")

    # Generate plots
    print("\nGenerating plots...")
    plot_prob_vs_age(df, str(output_dir / "prob_vs_age.png"))
    plot_mean_prob_by_age(df, args.age_bin_width, str(output_dir / "mean_prob_by_age.png"))
    plot_auroc_by_age(df, args.age_bin_width, str(output_dir / "auroc_by_age.png"))
    plot_per_site(df, str(output_dir / "per_site_age.png"))

    # Save raw data for later use
    df.to_csv(output_dir / "val_predictions.csv", index=False)
    print(f"  Saved {output_dir / 'val_predictions.csv'}")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
