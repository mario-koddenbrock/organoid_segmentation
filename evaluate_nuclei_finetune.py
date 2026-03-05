"""
Evaluate CellposeSAM (cpsam) on 3D organoid nuclei before and after finetuning.

The train/eval folder split and model paths are read from the same config file used
by finetune_nuclei.py, so both scripts share a single source of truth.
Fixed segmentation parameters (no hyperparameter optimization) are loaded from a
separate ModelConfig JSON.

Usage (from organoid_segmentation/):
    python evaluate_nuclei_finetune.py
    python evaluate_nuclei_finetune.py --finetune_config configs/finetune_nuclei_config.json
    python evaluate_nuclei_finetune.py --finetune_config configs/finetune_nuclei_config.json \\
                                       --output_dir results/my_comparison
"""

import argparse
import json
import logging
import os
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from cellpose_adapt import io
from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.core import CellposeRunner, initialize_model
from cellpose_adapt.logger import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.utils import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_finetune_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Finetune config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_data_sources(data_cfg: dict) -> list:
    """Return eval image directories directly from the finetune config."""
    return data_cfg["eval_image_dirs"]


def finetuned_model_path(ft_cfg: dict) -> str:
    """Derive the finetuned model path from the finetune config."""
    model_cfg = ft_cfg["model"]
    return os.path.join(model_cfg["save_dir"], "models", model_cfg["model_name"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    base_cfg: ModelConfig,
    model_name: str,
    data_sources: list,
    gt_mapping: dict,
    cache_dir: str,
    iou_threshold: float,
    device,
    label: str,
) -> list:
    """
    Run segmentation on all image/GT pairs and return per-image metric dicts.

    model_name overrides the model_name field in base_cfg so that the same
    segmentation parameters can be reused for both the pretrained and finetuned model.
    """
    cfg = replace(base_cfg, model_name=model_name)
    os.makedirs(cache_dir, exist_ok=True)

    data_pairs = io.find_image_gt_pairs(data_sources, gt_mapping, limit_per_source=None)
    if not data_pairs:
        logger.error("[%s] No image/GT pairs found. Check eval_folders in the finetune config.", label)
        return []

    logger.info("[%s] Found %d image/GT pairs. Starting evaluation...", label, len(data_pairs))

    model = initialize_model(cfg.model_name, device)
    runner = CellposeRunner(model, cfg, device, cache_dir=cache_dir)

    results = []
    for image_path, gt_path in data_pairs:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image, ground_truth, _ = io.load_image_with_gt(
            image_path, gt_path, cfg.channel_to_segment
        )

        if image is None:
            logger.warning("[%s] Could not load image: %s", label, image_path)
            continue

        pred_mask, duration = runner.run(image)

        if pred_mask is None:
            logger.warning("[%s] Prediction failed for: %s", label, base_name)
            continue

        if ground_truth is not None:
            stats = calculate_segmentation_stats(
                ground_truth, pred_mask, iou_threshold=iou_threshold
            )
            results.append({"image_name": base_name, "model": label, **stats})
            logger.info(
                "[%s] %s | Jaccard=%.4f | time=%.1fs",
                label, base_name, stats.get("jaccard", float("nan")), duration,
            )
        else:
            logger.info("[%s] %s | No ground truth available", label, base_name)

    logger.info("[%s] Evaluation done. %d images with metrics.", label, len(results))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_csv(results: list, path: str):
    if not results:
        logger.warning("No results to save to %s", path)
        return
    pd.DataFrame(results).to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved CSV: %s", path)


def _metric_columns(df: pd.DataFrame) -> list:
    skip = {"image_name", "model", "n_instances_true", "n_instances_pred"}
    return [c for c in df.columns if c not in skip]


def print_comparison_table(results_before: list, results_after: list):
    if not results_before or not results_after:
        logger.warning("Cannot print comparison: one or both result sets are empty.")
        return

    df_b = pd.DataFrame(results_before)
    df_a = pd.DataFrame(results_after)
    metric_cols = _metric_columns(df_b)

    logger.info("=" * 62)
    logger.info("Performance comparison (mean across all images):")
    logger.info("%-22s %12s %12s %10s", "Metric", "Pretrained", "Finetuned", "Delta")
    logger.info("-" * 58)
    for col in metric_cols:
        b = df_b[col].mean()
        a = df_a[col].mean()
        delta = a - b
        sign = "+" if delta >= 0 else ""
        logger.info("%-22s %12.4f %12.4f %s%.4f", col, b, a, sign, delta)
    logger.info("=" * 62)


def plot_comparison(results_before: list, results_after: list, output_path: str):
    """Grouped bar chart of mean metrics: pretrained vs. finetuned."""
    if not results_before or not results_after:
        logger.warning("Cannot create comparison plot: one or both result sets are empty.")
        return

    df_b = pd.DataFrame(results_before)
    df_a = pd.DataFrame(results_after)
    metric_cols = _metric_columns(df_b)

    means_before = df_b[metric_cols].mean()
    means_after = df_a[metric_cols].mean()

    x = np.arange(len(metric_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 2.5), 5))
    bars1 = ax.bar(x - width / 2, means_before.values, width, label="Pretrained (before)")
    bars2 = ax.bar(x + width / 2, means_after.values, width, label="Finetuned (after)")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Score")
    ax.set_title("Segmentation Performance: Pretrained vs. Finetuned CellposeSAM")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CellposeSAM before and after finetuning on organoid nuclei"
    )
    parser.add_argument(
        "--finetune_config",
        type=str,
        default="configs/finetune_nuclei_config.json",
        help="Finetune config (shared with finetune_nuclei.py). Contains all data, "
             "model, and evaluation settings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output_dir from the finetune config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: cuda / mps / cpu. Auto-detected if not specified.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(
        log_level=logging.INFO,
        log_file=f"eval_finetune_{timestamp}.log",
    )

    device = get_device(cli_device=args.device)
    logger.info("Using device: %s", device)

    # --- Load finetune config (single source of truth) ---
    ft_cfg = load_finetune_config(args.finetune_config)
    logger.info("Loaded finetune config from: %s", args.finetune_config)

    eval_cfg = ft_cfg["evaluation"]
    output_dir = args.output_dir or eval_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = ModelConfig.from_json(eval_cfg["model_config_path"])
    logger.info("Loaded segmentation config from: %s", eval_cfg["model_config_path"])

    # --- Data ---
    data_cfg = ft_cfg["data"]
    data_sources = build_data_sources(data_cfg)
    gt_mapping = data_cfg["gt_mapping"]
    logger.info("Eval dirs: %s", data_sources)

    # --- Model names ---
    pretrained_name = ft_cfg["model"]["base_model"]
    finetuned_name = finetuned_model_path(ft_cfg)
    logger.info("Pretrained model: %s", pretrained_name)
    logger.info("Finetuned model:  %s", finetuned_name)

    # --- Evaluate pretrained model ---
    logger.info("=== Evaluating PRETRAINED model ===")
    results_before = run_evaluation(
        model_cfg,
        model_name=pretrained_name,
        data_sources=data_sources,
        gt_mapping=gt_mapping,
        cache_dir=eval_cfg["cache_dir_pretrained"],
        iou_threshold=eval_cfg["iou_threshold"],
        device=device,
        label="pretrained",
    )
    save_csv(results_before, os.path.join(output_dir, "results_pretrained.csv"))

    # --- Evaluate finetuned model ---
    logger.info("=== Evaluating FINETUNED model ===")
    results_after = run_evaluation(
        model_cfg,
        model_name=finetuned_name,
        data_sources=data_sources,
        gt_mapping=gt_mapping,
        cache_dir=eval_cfg["cache_dir_finetuned"],
        iou_threshold=eval_cfg["iou_threshold"],
        device=device,
        label="finetuned",
    )
    save_csv(results_after, os.path.join(output_dir, "results_finetuned.csv"))

    # --- Compare ---
    print_comparison_table(results_before, results_after)
    plot_comparison(
        results_before,
        results_after,
        os.path.join(output_dir, "pretrained_vs_finetuned_comparison.png"),
    )

    logger.info("Evaluation complete. All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
