"""
Analyze and visualize hyperparameter search results.

Produces:
  1. ranked_trials.png      – top-20 trials by primary metric (bar chart)
  2. hparam_effects.png     – per-hyperparameter marginal effect (box plots)
  3. lr_wd_heatmap.png      – LR × WD heatmap, faceted by freeze_encoder
  4. 2d_vs_3d_scatter.png   – 2D AP@0.75 vs 3D AP@0.75, coloured by LR
  5. delta_from_baseline.png – improvement over the pretrained model
  6. training_time.png      – performance vs training time

Usage:
    python analyze_hparam_search.py [--results results/hparam_search/results.json]
                                    [--output  results/hparam_search]
                                    [--metric  3d_ap_0.75]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

THRESHOLDS = [0.5, 0.75, 0.9]
METRIC_COLS = (
    [f"2d_ap_{t}" for t in THRESHOLDS]
    + [f"3d_ap_{t}" for t in THRESHOLDS]
    + [f"delta_2d_ap_{t}" for t in THRESHOLDS]
    + [f"delta_3d_ap_{t}" for t in THRESHOLDS]
)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        if "error" in r:
            continue
        hp = r["hparams"]
        a2 = r.get("ap_after_2d", {})
        a3 = r.get("ap_after_3d", {})
        b2 = r.get("ap_before_2d", {})
        b3 = r.get("ap_before_3d", {})

        row = {
            "trial_idx": r["trial_idx"],
            "n_epochs": hp["n_epochs"],
            "learning_rate": hp["learning_rate"],
            "weight_decay": hp["weight_decay"],
            "freeze_encoder": hp["freeze_encoder"],
            "batch_size": hp.get("batch_size", 8),
            "train_loss": r.get("train_loss_final", float("nan")),
            "train_time_s": r.get("train_time_s", float("nan")),
        }
        for t in THRESHOLDS:
            row[f"2d_ap_{t}"] = float(a2.get(str(t), a2.get(t, float("nan"))))
            row[f"3d_ap_{t}"] = float(a3.get(str(t), a3.get(t, float("nan"))))
            row[f"delta_2d_ap_{t}"] = row[f"2d_ap_{t}"] - float(b2.get(str(t), b2.get(t, 0)))
            row[f"delta_3d_ap_{t}"] = row[f"3d_ap_{t}"] - float(b3.get(str(t), b3.get(t, 0)))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["lr_label"] = df["learning_rate"].map(lambda x: f"{x:.0e}")
    df["wd_label"] = df["weight_decay"].map(lambda x: f"{x:.0e}")
    df["frozen_label"] = df["freeze_encoder"].map({True: "frozen", False: "unfrozen"})
    return df


def metric_col(metric_key: str) -> str:
    """Convert '3d_ap_0.75' -> '3d_ap_0.75' (pass-through; validates key exists)."""
    parts = metric_key.split("_")
    # e.g. "3d_ap_0.75" -> dim="3d", threshold=0.75
    return metric_key


# ---------------------------------------------------------------------------
# Plot 1: Ranked trials
# ---------------------------------------------------------------------------

def plot_ranked_trials(df: pd.DataFrame, metric: str, out_path: str, top_n: int = 20,
                       companion: str | None = None):
    has_companion = companion is not None and companion != metric and companion in df.columns

    ranked = df.nlargest(min(top_n, len(df)), metric)
    n = len(ranked)
    x = np.arange(n)
    width = 0.55 if not has_companion else 0.38

    fig, ax = plt.subplots(figsize=(max(12, n * 0.6), 5.5))

    colors = ["#1f77b4" if frz else "#ff7f0e" for frz in ranked["freeze_encoder"]]

    if has_companion:
        bars_primary = ax.bar(x - width / 2, ranked[metric],    width, color=colors,
                              edgecolor="white", linewidth=0.5, label=metric, zorder=3)
        bars_companion = ax.bar(x + width / 2, ranked[companion], width,
                                color=[c + "88" for c in  # same hue, lighter via alpha trick
                                       ["#1f77b4" if frz else "#ff7f0e" for frz in ranked["freeze_encoder"]]],
                                edgecolor="white", linewidth=0.5, label=companion, zorder=3,
                                hatch="///")
        all_bar_groups = [(bars_primary, metric), (bars_companion, companion)]
    else:
        bars_primary = ax.bar(x, ranked[metric], width, color=colors,
                              edgecolor="white", linewidth=0.5, zorder=3)
        all_bar_groups = [(bars_primary, metric)]

    # Value labels on top of each bar
    for bars, _ in all_bar_groups:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    # Trial index below x-axis (via xticklabels)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"#{r.trial_idx}\nlr={r.lr_label}\nwd={r.wd_label}\nep={r.n_epochs}"
        for r in ranked.itertuples()
    ], fontsize=7)

    # Baseline reference lines
    for m, ls, lw in [(metric, "--", 1.4), (companion, ":", 1.0)] if has_companion else [(metric, "--", 1.4)]:
        delta_col = f"delta_{m}"
        if delta_col in df.columns:
            baseline = float(df[m].iloc[0] - df[delta_col].iloc[0])
            ax.axhline(baseline, color="red", linestyle=ls, linewidth=lw,
                       alpha=0.7, label=f"Baseline {m} ({baseline:.3f})", zorder=2)

    ax.set_ylabel("AP")
    ax.set_title(f"Top {n} trials ranked by {metric}" + (f"  |  also showing {companion}" if has_companion else ""))
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#1f77b4", label="frozen encoder"),
        Patch(color="#ff7f0e", label="unfrozen encoder"),
    ]
    if has_companion:
        legend_handles += [
            Patch(facecolor="grey", label=metric, zorder=3),
            Patch(facecolor="grey", hatch="///", label=companion, zorder=3),
        ]
    legend_handles += ax.get_lines()
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-hyperparameter marginal effects (box plots)
# ---------------------------------------------------------------------------

def plot_hparam_effects(df: pd.DataFrame, metric: str, out_path: str):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"Marginal effect of each hyperparameter on {metric}", y=1.01)

    hparams = [
        ("learning_rate", "lr_label", "Learning rate"),
        ("weight_decay",  "wd_label",  "Weight decay"),
        ("n_epochs",      "n_epochs",  "Epochs"),
        ("freeze_encoder","frozen_label","Encoder"),
    ]

    for ax, (col, label_col, title) in zip(axes, hparams):
        groups = df.groupby(label_col)[metric].apply(list)
        labels = list(groups.index)
        data = list(groups.values)

        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="red", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec6cf")
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(metric if ax == axes[0] else "")
        ax.set_ylim(bottom=max(0, df[metric].min() - 0.05))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: LR × WD heatmap (faceted by freeze_encoder)
# ---------------------------------------------------------------------------

def plot_lr_wd_heatmap(df: pd.DataFrame, metric: str, out_path: str):
    frozen_vals = df["freeze_encoder"].unique()
    n_cols = len(frozen_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    lr_vals = sorted(df["learning_rate"].unique())
    wd_vals = sorted(df["weight_decay"].unique())
    lr_labels = [f"{v:.0e}" for v in lr_vals]
    wd_labels = [f"{v:.0e}" for v in wd_vals]

    vmin = df[metric].min()
    vmax = df[metric].max()

    for ax, frz in zip(axes, sorted(frozen_vals, reverse=True)):
        sub = df[df["freeze_encoder"] == frz]
        # Average over n_epochs (shows best achievable for each LR/WD combo)
        pivot = sub.groupby(["learning_rate", "weight_decay"])[metric].max().unstack("weight_decay")
        pivot = pivot.reindex(index=lr_vals, columns=wd_vals)

        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax, cmap="YlGn")
        ax.set_xticks(range(len(wd_vals)))
        ax.set_xticklabels(wd_labels, fontsize=8)
        ax.set_yticks(range(len(lr_vals)))
        ax.set_yticklabels(lr_labels, fontsize=8)
        ax.set_xlabel("Weight decay")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{'Frozen' if frz else 'Unfrozen'} encoder\n(max over epochs)")

        for i in range(len(lr_vals)):
            for j in range(len(wd_vals)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                            color="black" if val < (vmin + vmax) / 2 else "white")

        fig.colorbar(im, ax=ax, label=metric, fraction=0.046, pad=0.04)

    fig.suptitle(f"LR × WD heatmap — {metric} (max over epochs)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: 2D vs 3D scatter
# ---------------------------------------------------------------------------

def plot_2d_vs_3d(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    lr_vals = sorted(df["learning_rate"].unique())
    cmap = plt.cm.get_cmap("tab10", len(lr_vals))
    lr_color = {lr: cmap(i) for i, lr in enumerate(lr_vals)}

    for _, row in df.iterrows():
        color = lr_color[row["learning_rate"]]
        marker = "o" if row["freeze_encoder"] else "^"
        ax.scatter(row["2d_ap_0.75"], row["3d_ap_0.75"],
                   color=color, marker=marker, s=60, alpha=0.75, edgecolors="white", linewidths=0.5)

    # Baseline cross
    baseline_2d = float(df["2d_ap_0.75"].iloc[0] - df["delta_2d_ap_0.75"].iloc[0])
    baseline_3d = float(df["3d_ap_0.75"].iloc[0] - df["delta_3d_ap_0.75"].iloc[0])
    ax.axvline(baseline_2d, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(baseline_3d, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(baseline_2d + 0.003, ax.get_ylim()[0] + 0.01, "baseline", color="red", fontsize=7)

    # Legends
    from matplotlib.lines import Line2D
    legend_lr = [Line2D([0], [0], marker="o", color="w", markerfacecolor=lr_color[lr],
                        markersize=8, label=f"lr={lr:.0e}") for lr in lr_vals]
    legend_frz = [
        Line2D([0], [0], marker="o", color="grey", markersize=8, label="frozen"),
        Line2D([0], [0], marker="^", color="grey", markersize=8, label="unfrozen"),
    ]
    ax.legend(handles=legend_lr + legend_frz, fontsize=8, loc="lower right")

    ax.set_xlabel("2D AP@0.75")
    ax.set_ylabel("3D AP@0.75")
    ax.set_title("2D vs 3D performance trade-off  (marker = frozen○ / unfrozen△,  color = LR)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Delta from baseline
# ---------------------------------------------------------------------------

def plot_deltas(df: pd.DataFrame, out_path: str):
    metrics = [
        ("delta_2d_ap_0.5",  "2D AP@0.50"),
        ("delta_2d_ap_0.75", "2D AP@0.75"),
        ("delta_2d_ap_0.9",  "2D AP@0.90"),
        ("delta_3d_ap_0.5",  "3D AP@0.50"),
        ("delta_3d_ap_0.75", "3D AP@0.75"),
        ("delta_3d_ap_0.9",  "3D AP@0.90"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Improvement over pretrained baseline (Δ AP)", y=1.01)

    for ax, (col, title) in zip(axes.flat, metrics):
        sorted_df = df.sort_values(col, ascending=True)
        colors = ["#ff7f0e" if not frz else "#1f77b4" for frz in sorted_df["freeze_encoder"]]
        bars = ax.barh(range(len(sorted_df)), sorted_df[col], color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xlabel("Δ AP")
        ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color="#1f77b4", label="frozen encoder"),
        Patch(color="#ff7f0e", label="unfrozen encoder"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 6: Performance vs training time
# ---------------------------------------------------------------------------

def plot_time_vs_perf(df: pd.DataFrame, metric: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    frozen_vals = [True, False]
    markers = {True: "o", False: "^"}
    colors = {True: "#1f77b4", False: "#ff7f0e"}

    for frz in frozen_vals:
        sub = df[df["freeze_encoder"] == frz]
        ax.scatter(sub["train_time_s"] / 60, sub[metric],
                   marker=markers[frz], color=colors[frz], s=60,
                   alpha=0.8, edgecolors="white", linewidths=0.5,
                   label=f"{'frozen' if frz else 'unfrozen'} encoder")

    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs training time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, metric: str, top_n: int = 10):
    ranked = df.nlargest(top_n, metric)
    cols = ["trial_idx", "freeze_encoder", "learning_rate", "weight_decay", "n_epochs",
            "3d_ap_0.5", "3d_ap_0.75", "3d_ap_0.9",
            "2d_ap_0.5", "2d_ap_0.75", "2d_ap_0.9",
            "train_loss", "train_time_s"]
    cols = [c for c in cols if c in df.columns]

    print(f"\n{'='*110}")
    print(f"TOP {top_n} TRIALS  (ranked by {metric})")
    print("=" * 110)
    print(ranked[cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 110)

    best = ranked.iloc[0]
    print(f"\nBest trial #{int(best['trial_idx'])}:")
    print(f"  learning_rate : {best['learning_rate']:.1e}")
    print(f"  weight_decay  : {best['weight_decay']:.1e}")
    print(f"  n_epochs      : {int(best['n_epochs'])}")
    print(f"  freeze_encoder: {best['freeze_encoder']}")
    print(f"  {metric}     : {best[metric]:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("--results", default="results/hparam_search/results.json")
    parser.add_argument("--output",  default="results/hparam_search")
    parser.add_argument("--metric",  default="3d_ap_0.75",
                        help="Primary metric for ranking (e.g. 3d_ap_0.75, 2d_ap_0.75)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = load_results(args.results)
    print(f"Loaded {len(df)} valid trials from {args.results}")

    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {[c for c in df.columns if 'ap' in c]}")

    print_summary(df, metric)

    # Ranked-trials plots for all 3D metrics with explicit companion pairings
    ranked_3d_configs = [
        ("3d_ap_0.5",  "3d_ap_0.75"),
        ("3d_ap_0.75", "3d_ap_0.9"),
        ("3d_ap_0.9",  "3d_ap_0.75"),
    ]
    for primary, companion in ranked_3d_configs:
        fname = f"ranked_trials_{primary.replace('.', '_')}.png"
        plot_ranked_trials(df, primary, os.path.join(args.output, fname), companion=companion)

    plot_hparam_effects(df, metric,
                        os.path.join(args.output, "hparam_effects.png"))
    plot_lr_wd_heatmap(df, metric,
                       os.path.join(args.output, "lr_wd_heatmap.png"))
    plot_2d_vs_3d(df,
                  os.path.join(args.output, "2d_vs_3d_scatter.png"))
    plot_deltas(df,
                os.path.join(args.output, "delta_from_baseline.png"))
    plot_time_vs_perf(df, metric,
                      os.path.join(args.output, "training_time.png"))

    print(f"\nAll plots saved to: {args.output}/")


if __name__ == "__main__":
    main()
