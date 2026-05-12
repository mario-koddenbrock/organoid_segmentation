"""
Run predictions with the best finetuned models on all organoid datasets
and evaluate against ground-truth labels where available.

For each of the image directories configured in predict_dirs:
  - Runs 3D nuclei prediction, saves masks
  - If ground-truth labels exist (Organoids folder), computes per-image AP@0.50/0.75/0.90

All outputs are written to the unified results folder:
    results/predictions/<model_name>/<group>/<date_folder>/*.tif
    results/reports/evaluation_summary.json
    results/reports/evaluation_summary.txt

Usage:
    python evaluate_all.py --config configs/predict_all_cluster_config.json [--device cuda]
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
from cellpose import metrics as cpm
from cellpose import models
from tifffile import imread, imwrite

from finetune_nuclei import (
    _derive_gt_path,
    extract_nuclei_channel,
    get_device,
)
from predict_akps_nuclei import collect_explicit_dirs, predict_volume

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("cellpose").setLevel(logging.WARNING)

THRESHOLDS = [0.5, 0.75, 0.9]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-directory prediction + evaluation
# ---------------------------------------------------------------------------

def process_directory(
    model,
    image_dir: str,
    out_dir: str,
    gt_mapping,
) -> list:
    """
    Predict all .tif files in image_dir; evaluate against GT where it exists.
    Returns a list of per-image result dicts.
    """
    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff")))
    if not image_files:
        logging.warning(f"No .tif files in {image_dir}, skipping.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for i, fname in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, fname)
        out_path = os.path.join(out_dir, fname)
        entry: dict = {"image": fname, "has_gt": False}

        # Predict (skip if mask already exists)
        if os.path.exists(out_path):
            logging.info(f"  [{i}/{len(image_files)}] Already exists, loading mask: {fname}")
            pred_mask = imread(out_path)
        else:
            try:
                volume = imread(img_path)
            except Exception as e:
                logging.warning(f"  [{i}/{len(image_files)}] Unreadable: {fname} — {e}")
                results.append(entry)
                continue
            nuclei_vol = extract_nuclei_channel(volume)
            logging.info(f"  [{i}/{len(image_files)}] {fname}  shape={nuclei_vol.shape}")
            pred_mask = predict_volume(model, nuclei_vol)
            imwrite(out_path, pred_mask, compression="zlib")

        # Evaluate if GT label exists
        if gt_mapping:
            gt_path = _derive_gt_path(img_path, gt_mapping)
            if os.path.isfile(gt_path):
                gt_mask = imread(gt_path)
                ap, _, _, _ = cpm.average_precision([gt_mask], [pred_mask], threshold=THRESHOLDS)
                entry["has_gt"] = True
                entry["ap_3d"] = {str(t): float(ap[0, idx]) for idx, t in enumerate(THRESHOLDS)}
                logging.info(
                    f"    GT: AP@0.50={entry['ap_3d']['0.5']:.4f}  "
                    f"AP@0.75={entry['ap_3d']['0.75']:.4f}  "
                    f"AP@0.9={entry['ap_3d']['0.9']:.4f}"
                )

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_reports(all_results: dict, reports_dir: str):
    """Write evaluation_summary.json and evaluation_summary.txt."""
    os.makedirs(reports_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(reports_dir, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"JSON report: {json_path}")

    # Text
    txt_path = os.path.join(reports_dir, "evaluation_summary.txt")
    lines = [
        "=" * 80,
        f"ORGANOID NUCLEI SEGMENTATION — EVALUATION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
    ]

    for model_name, model_data in all_results.items():
        lines += [f"MODEL: {model_name}", "-" * 60]
        if model_data.get("hparams"):
            hp = model_data["hparams"]
            lines.append(f"  lr={hp.get('learning_rate', '?'):.1e}  "
                         f"wd={hp.get('weight_decay', '?'):.1e}  "
                         f"epochs={hp.get('n_epochs', '?')}  "
                         f"frozen={hp.get('freeze_encoder', '?')}")
        lines.append("")

        # Aggregate eval metrics
        eval_images = [
            img for dir_data in model_data.get("dirs", {}).values()
            for img in dir_data
            if img.get("has_gt")
        ]
        if eval_images:
            for th in THRESHOLDS:
                vals = [img["ap_3d"][str(th)] for img in eval_images if str(th) in img.get("ap_3d", {})]
                mean_ap = sum(vals) / len(vals) if vals else float("nan")
                lines.append(f"  Mean 3D AP@{th:.2f} = {mean_ap:.4f}  (n={len(vals)})")
            lines.append("")

        # Per-dir breakdown
        lines.append("  Per-directory breakdown:")
        for dir_key, dir_images in model_data.get("dirs", {}).items():
            gt_imgs = [img for img in dir_images if img.get("has_gt")]
            no_gt = len(dir_images) - len(gt_imgs)
            if gt_imgs:
                mean_075 = sum(img["ap_3d"]["0.75"] for img in gt_imgs) / len(gt_imgs)
                lines.append(f"    {dir_key}: {len(gt_imgs)} eval, mean AP@0.75={mean_075:.4f}  ({no_gt} no-GT)")
            else:
                lines.append(f"    {dir_key}: {len(dir_images)} images, no GT")
        lines.append("")

    lines.append("=" * 80)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"Text report: {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best models on all organoid data")
    parser.add_argument("--config", default="configs/predict_all_cluster_config.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = get_device() if args.device is None else __import__("torch").device(args.device)
    logging.info(f"Using device: {device}")

    cfg = load_config(args.config)
    model_configs = cfg["models"]
    models_dir = cfg["output"]["models_dir"]
    predictions_dir = cfg["output"]["predictions_dir"]
    reports_dir = cfg["output"]["reports_dir"]
    gt_mapping = cfg.get("gt_mapping")

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    use_bfloat16 = device.type == "cuda"

    # Discover all image directories
    leaf_dirs = collect_explicit_dirs(cfg["predict_dirs"])
    total_images = sum(
        len([f for f in os.listdir(p) if f.lower().endswith((".tif", ".tiff"))])
        for _, _, p in leaf_dirs
    )
    logging.info(f"Found {len(leaf_dirs)} image directories, {total_images} images total")

    all_results = {
        "generated": datetime.now().isoformat(),
        "models": {},
    }

    for m in model_configs:
        model_name = m["name"]
        model_path = os.path.join(models_dir, "models", model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating: {model_name}")
        logging.info(f"{'='*60}")

        cpsam_model = models.CellposeModel(
            pretrained_model=model_path, device=device, use_bfloat16=use_bfloat16,
        )

        model_results: dict = {"hparams": m, "dirs": {}}

        for group, date_folder, image_dir in leaf_dirs:
            dir_key = f"{group}/{date_folder}"
            out_dir = os.path.join(predictions_dir, model_name, group, date_folder)
            logging.info(f"\n  {dir_key}")

            dir_results = process_directory(cpsam_model, image_dir, out_dir, gt_mapping)
            model_results["dirs"][dir_key] = dir_results

        all_results["models"][model_name] = model_results
        del cpsam_model

    write_reports(all_results, reports_dir)
    logging.info(f"\nAll done. Results in: {os.path.dirname(predictions_dir)}")


if __name__ == "__main__":
    main()
