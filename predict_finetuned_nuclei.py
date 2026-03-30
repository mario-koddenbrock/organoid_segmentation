"""
Retrain the two best trials from the hyperparameter search, then predict
nuclei instance masks for every organoid image across all data directories.

Steps:
  1. Load training/eval data (same split as finetuning).
  2. For each trial config, train a CellposeSAM model and save it.
  3. For each saved model, run 3D inference on every .tif in predict_dirs
     and write the predicted mask to a per-trial output folder, mirroring
     the source directory name.

Predicted masks are saved as uint32 .tif files with the same filename as
the input image but in:
  <predictions_dir>/<trial_name>/<source_dataset_name>/<image_name>

Usage:
    python predict_finetuned_nuclei.py \
        [--config configs/predict_finetuned_nuclei_cluster_config.json] \
        [--device cuda] \
        [--skip-training]   # if models are already trained
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
from cellpose import models, train
from tifffile import imread, imwrite

from finetune_nuclei import (
    collect_image_mask_pairs,
    evaluate_model,
    evaluate_model_3d,
    extract_nuclei_channel,
    get_device,
    load_and_slice_pairs,
    load_finetune_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_trial(trial: dict, base_cfg: dict, train_images, train_masks,
                eval_images, eval_masks, models_dir: str, device) -> str:
    """Train one trial and return the path to the saved model."""
    model_cfg = base_cfg["model"]
    training_cfg = base_cfg["training"]
    use_bfloat16 = device.type == "cuda"

    logging.info(f"\n{'='*60}")
    logging.info(f"Training: {trial['name']}")
    logging.info(f"  lr={trial['learning_rate']:.1e}  wd={trial['weight_decay']:.1e}  "
                 f"epochs={trial['n_epochs']}  frozen={trial['freeze_encoder']}")
    logging.info(f"{'='*60}")

    model = models.CellposeModel(
        pretrained_model=model_cfg["base_model"],
        device=device,
        use_bfloat16=use_bfloat16,
    )

    if trial["freeze_encoder"]:
        for param in model.net.encoder.parameters():
            param.requires_grad = False
        logging.info("Encoder frozen")

    nimg_per_epoch = min(800, max(8, len(train_images)))

    model_path, _, _ = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=eval_images or None,
        test_labels=eval_masks or None,
        n_epochs=trial["n_epochs"],
        learning_rate=trial["learning_rate"],
        batch_size=trial["batch_size"],
        weight_decay=trial["weight_decay"],
        normalize=True,
        save_path=models_dir,
        save_every=trial["n_epochs"],
        save_each=False,
        min_train_masks=training_cfg["min_train_masks"],
        model_name=trial["name"],
        scale_range=training_cfg["scale_range"],
        bsize=training_cfg["bsize"],
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=len(eval_images) if eval_images else None,
    )

    logging.info(f"Model saved: {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

THRESHOLDS = (0.5, 0.75, 0.9)
TOLERANCE = 0.02  # absolute AP difference flagged as a deviation


def write_report(trial: dict, model_path: str,
                 eval_images, eval_masks, eval_volumes,
                 report_path: str, device):
    """
    Evaluate the retrained model on the eval set, compare to expected metrics
    from the hparam search, and write a Markdown report to report_path.
    """
    use_bfloat16 = device.type == "cuda"
    model = models.CellposeModel(
        pretrained_model=model_path, device=device, use_bfloat16=use_bfloat16,
    )

    logging.info(f"Evaluating {trial['name']} for validation report...")
    ap_2d, _ = evaluate_model(model, eval_images, eval_masks, thresholds=THRESHOLDS)
    ap_3d, _ = evaluate_model_3d(model, eval_volumes, thresholds=THRESHOLDS)

    expected = trial.get("expected_metrics", {})
    exp_2d = expected.get("ap_after_2d", {})
    exp_3d = expected.get("ap_after_3d", {})

    lines = [
        f"# Validation Report — {trial['name']}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Model: `{model_path}`  ",
        f"",
        f"## Hyperparameters",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| learning_rate | `{trial['learning_rate']:.1e}` |",
        f"| weight_decay | `{trial['weight_decay']:.1e}` |",
        f"| n_epochs | `{trial['n_epochs']}` |",
        f"| freeze_encoder | `{trial['freeze_encoder']}` |",
        f"| batch_size | `{trial['batch_size']}` |",
        f"",
        f"## Results vs Expected",
        f"",
        f"| Metric | Expected | Achieved | Δ | Status |",
        f"|--------|----------|----------|---|--------|",
    ]

    all_ok = True
    for label, achieved, expected_dict in [("2D", ap_2d, exp_2d), ("3D", ap_3d, exp_3d)]:
        for t in THRESHOLDS:
            ach = achieved.get(t, float("nan"))
            exp = float(expected_dict.get(str(t), expected_dict.get(t, float("nan"))))
            if np.isnan(exp):
                status = "⚠ no reference"
                delta_str = "—"
            else:
                delta = ach - exp
                delta_str = f"{delta:+.4f}"
                if abs(delta) > TOLERANCE:
                    status = f"❌ deviation > {TOLERANCE}"
                    all_ok = False
                else:
                    status = "✅ OK"
            exp_str = f"{exp:.4f}" if not np.isnan(exp) else "—"
            lines.append(f"| {label} AP@{t:.2f} | {exp_str} | {ach:.4f} | {delta_str} | {status} |")

    lines += [
        f"",
        f"## Summary",
        f"",
        f"{'✅ All metrics within tolerance.' if all_ok else f'❌ Some metrics deviate by more than {TOLERANCE}. Check training stability.'}",
        f"",
        f"Tolerance used: ±{TOLERANCE} absolute AP.",
    ]

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logging.info(f"Validation report written: {report_path}")
    if not all_ok:
        logging.warning(f"Some metrics deviate from the hparam search reference by more than {TOLERANCE}!")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_volume(model, nuclei_vol: np.ndarray) -> np.ndarray:
    """Run 3D inference on a single nuclei volume (Z, H, W). Returns label mask."""
    img = np.stack(
        (nuclei_vol, np.zeros_like(nuclei_vol), np.zeros_like(nuclei_vol)),
        axis=-1,
    )
    mask, _, _ = model.eval(
        img, channel_axis=-1, z_axis=0,
        normalize=True,
        do_3D=True, flow3D_smooth=2,
        diameter=30, bsize=256, batch_size=64,
        niter=1000, min_size=1000,
    )
    return mask.astype(np.uint32)


def predict_directory(model, image_dir: str, out_dir: str):
    """Run inference on all .tif files in image_dir and save masks to out_dir."""
    image_files = sorted([
        f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))
    ])
    if not image_files:
        logging.warning(f"No .tif files found in {image_dir}, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Predicting {len(image_files)} images from: {image_dir}")
    logging.info(f"  -> Output: {out_dir}")

    for i, fname in enumerate(image_files, 1):
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            logging.info(f"  [{i}/{len(image_files)}] Skipping (already exists): {fname}")
            continue

        img_path = os.path.join(image_dir, fname)
        volume = imread(img_path)
        nuclei_vol = extract_nuclei_channel(volume)

        logging.info(f"  [{i}/{len(image_files)}] {fname}  shape={nuclei_vol.shape}")
        mask = predict_volume(model, nuclei_vol)
        imwrite(out_path, mask, compression="zlib")

    logging.info(f"Done: {image_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain best hparam trials and predict nuclei masks")
    parser.add_argument("--config", default="configs/predict_finetuned_nuclei_cluster_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and load existing model files from models_dir")
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
    base_cfg = load_finetune_config(cfg["base_config"])

    trials = cfg["trials"]
    predict_dirs = cfg["predict_dirs"]
    models_dir = cfg["output"]["models_dir"]
    predictions_dir = cfg["output"]["predictions_dir"]

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load training data (shared across trials)
    # ------------------------------------------------------------------
    if not args.skip_training:
        data_cfg = base_cfg["data"]
        training_cfg = base_cfg["training"]
        axis_map = {"Z": 0, "Y": 1, "X": 2}
        slice_axes = tuple(axis_map[a.upper()] for a in training_cfg.get("slice_axes", ["Z"]))

        logging.info("Collecting training pairs...")
        train_pairs = collect_image_mask_pairs(data_cfg["train_image_dirs"], data_cfg["gt_mapping"])
        logging.info(f"Found {len(train_pairs)} training pairs")

        logging.info("Collecting eval pairs...")
        eval_pairs = collect_image_mask_pairs(data_cfg["eval_image_dirs"], data_cfg["gt_mapping"])

        logging.info("Loading and slicing training data...")
        train_images, train_masks, _ = load_and_slice_pairs(
            train_pairs,
            min_masks_per_slice=training_cfg["min_masks_per_slice"],
            min_pixels=training_cfg["min_pixels"],
            slice_axes=slice_axes,
        )
        logging.info(f"Training set: {len(train_images)} slices")

        logging.info("Loading and slicing eval data...")
        eval_images, eval_masks, eval_volumes = load_and_slice_pairs(
            eval_pairs,
            min_masks_per_slice=training_cfg["min_masks_per_slice"],
            min_pixels=training_cfg["min_pixels"],
            slice_axes=slice_axes,
        )
        logging.info(f"Eval set: {len(eval_images)} slices, {len(eval_volumes)} 3D volumes")

    # ------------------------------------------------------------------
    # 2. Train each trial (or locate existing model)
    # ------------------------------------------------------------------
    use_bfloat16 = device.type == "cuda"
    trial_model_paths = {}

    reports_dir = cfg["output"].get("reports_dir",
                  os.path.join(os.path.dirname(models_dir), "reports"))

    for trial in trials:
        expected_path = os.path.join(models_dir, "models", trial["name"])

        if args.skip_training or os.path.exists(expected_path):
            if not os.path.exists(expected_path):
                raise FileNotFoundError(
                    f"--skip-training set but model not found: {expected_path}"
                )
            logging.info(f"Loading existing model: {expected_path}")
            model_path = expected_path
        else:
            model_path = train_trial(
                trial, base_cfg,
                train_images, train_masks,
                eval_images, eval_masks,
                models_dir, device,
            )

        trial_model_paths[trial["name"]] = model_path

        # Write validation report immediately after training
        if not args.skip_training:
            report_path = os.path.join(reports_dir, f"{trial['name']}_validation.md")
            write_report(trial, model_path, eval_images, eval_masks, eval_volumes,
                         report_path, device)

    # ------------------------------------------------------------------
    # 3. Predict on all data directories
    # ------------------------------------------------------------------
    for trial in trials:
        trial_name = trial["name"]
        model_path = trial_model_paths[trial_name]
        logging.info(f"\nRunning predictions with model: {trial_name}")

        model = models.CellposeModel(
            pretrained_model=model_path,
            device=device,
            use_bfloat16=use_bfloat16,
        )

        for image_dir in predict_dirs:
            if not os.path.isdir(image_dir):
                logging.warning(f"Directory not found, skipping: {image_dir}")
                continue

            # Preserve the dataset name as a subfolder
            dataset_name = os.path.basename(image_dir.rstrip("/"))
            out_dir = os.path.join(predictions_dir, trial_name, dataset_name)
            predict_directory(model, image_dir, out_dir)

        logging.info(f"All predictions for {trial_name} written to: {predictions_dir}/{trial_name}/")

    logging.info("\nDone.")


if __name__ == "__main__":
    main()