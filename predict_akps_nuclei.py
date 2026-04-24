"""
Predict nuclei instance masks on the AKPS progression dataset using the
pretrained CellposeSAM model and the best finetuned models from the
hyperparameter search.

For finetuned models the weights are retrained from scratch (the hparam
search deleted checkpoints after evaluation). Already-trained model files
and already-predicted mask files are skipped, so the job is safe to resubmit.

Directory structure expected under data_root:
    <data_root>/<group>/<date_folder>/*.tif
    e.g. AKPS_Progression_Organoids/AKP/20260226_AKP_images_rescaled_cropped/image.tif

Output masks mirror the source tree under predictions_dir:
    <predictions_dir>/<model_name>/<group>/<date_folder>/<image_name>.tif

Usage:
    python predict_akps_nuclei.py [--config configs/predict_akps_cluster_config.json]
                                  [--device cuda]
                                  [--skip-training]
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

def train_model(model_cfg: dict, base_cfg: dict, train_images, train_masks,
                eval_images, eval_masks, models_dir: str, device) -> str:
    """Finetune one model config and return the saved model path."""
    training_cfg = base_cfg["training"]
    use_bfloat16 = device.type == "cuda"

    logging.info(f"\n{'='*60}")
    logging.info(f"Training: {model_cfg['name']}")
    logging.info(f"  lr={model_cfg['learning_rate']:.1e}  wd={model_cfg['weight_decay']:.1e}  "
                 f"epochs={model_cfg['n_epochs']}  frozen={model_cfg['freeze_encoder']}")
    logging.info(f"{'='*60}")

    model = models.CellposeModel(
        pretrained_model=model_cfg["base_model"],
        device=device,
        use_bfloat16=use_bfloat16,
    )

    if model_cfg["freeze_encoder"]:
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
        n_epochs=model_cfg["n_epochs"],
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        weight_decay=model_cfg["weight_decay"],
        normalize=True,
        save_path=models_dir,
        save_every=model_cfg["n_epochs"],
        save_each=False,
        min_train_masks=training_cfg["min_train_masks"],
        model_name=model_cfg["name"],
        scale_range=training_cfg["scale_range"],
        bsize=training_cfg["bsize"],
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=len(eval_images) if eval_images else None,
    )

    logging.info(f"Model saved: {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_volume(model, nuclei_vol: np.ndarray) -> np.ndarray:
    """Run 3D inference on a single nuclei volume (Z, H, W). Returns uint32 label mask."""
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
    """Predict all .tif files in image_dir and write masks to out_dir."""
    image_files = sorted([
        f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))
    ])
    if not image_files:
        logging.warning(f"No .tif files in {image_dir}, skipping.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    n_done = 0

    for i, fname in enumerate(image_files, 1):
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            logging.info(f"  [{i}/{len(image_files)}] Already exists, skipping: {fname}")
            continue

        try:
            volume = imread(os.path.join(image_dir, fname))
        except Exception as e:
            logging.warning(f"  [{i}/{len(image_files)}] Skipping (unreadable): {fname} — {e}")
            continue
        nuclei_vol = extract_nuclei_channel(volume)
        logging.info(f"  [{i}/{len(image_files)}] {fname}  shape={nuclei_vol.shape}")

        mask = predict_volume(model, nuclei_vol)
        imwrite(out_path, mask, compression="zlib")
        n_done += 1

    return n_done


def collect_explicit_dirs(predict_dirs: list[str]) -> list[tuple[str, str, str]]:
    """Convert a flat list of absolute paths to (parent_folder, subdir, abs_path) tuples."""
    result = []
    for path in predict_dirs:
        path = os.path.abspath(path)
        subdir = os.path.basename(path)
        parent = os.path.basename(os.path.dirname(path))
        has_tifs = os.path.isdir(path) and any(
            f.lower().endswith((".tif", ".tiff")) for f in os.listdir(path)
        )
        if has_tifs:
            result.append((parent, subdir, path))
        else:
            logging.warning(f"No .tif files found (skipping): {path}")
    return result


def collect_leaf_dirs(data_root: str) -> list[tuple[str, str, str]]:
    """
    Walk data_root two levels deep and return (group, date_folder, abs_path)
    for every leaf directory that contains at least one .tif file.
    Skips the predictions/ subfolder.
    """
    leaf_dirs = []
    for group in sorted(os.listdir(data_root)):
        if group == "predictions":
            continue
        group_path = os.path.join(data_root, group)
        if not os.path.isdir(group_path):
            continue
        for date_folder in sorted(os.listdir(group_path)):
            folder_path = os.path.join(group_path, date_folder)
            if not os.path.isdir(folder_path):
                continue
            has_tifs = any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(folder_path))
            if has_tifs:
                leaf_dirs.append((group, date_folder, folder_path))
    return leaf_dirs


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(model_cfg: dict, leaf_dirs: list, predictions_dir: str, report_dir: str):
    """Write a Markdown summary including finetuning parameters and per-folder prediction counts."""
    os.makedirs(report_dir, exist_ok=True)
    model_name = model_cfg["name"]
    report_path = os.path.join(report_dir, f"{model_name}_prediction_summary.md")

    lines = [
        f"# Prediction Summary — {model_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]

    # Finetuning parameters section
    if model_cfg.get("skip_training", False):
        lines += [
            f"## Model",
            f"",
            f"**Pretrained CellposeSAM (`cpsam`) — no finetuning.**  ",
            f"Used as baseline for comparison.",
            f"",
        ]
    else:
        lines += [
            f"## Finetuning Parameters",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Base model | `{model_cfg['base_model']}` |",
            f"| Learning rate | `{model_cfg['learning_rate']:.1e}` |",
            f"| Weight decay | `{model_cfg['weight_decay']:.1e}` |",
            f"| Epochs | `{model_cfg['n_epochs']}` |",
            f"| Freeze encoder | `{model_cfg['freeze_encoder']}` |",
            f"| Batch size | `{model_cfg['batch_size']}` |",
            f"",
        ]

        # Expected metrics from hparam search if present
        exp = model_cfg.get("expected_metrics", {})
        if exp:
            e2 = exp.get("ap_after_2d", {})
            e3 = exp.get("ap_after_3d", {})
            lines += [
                f"## Expected Metrics (from hyperparameter search)",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
            ]
            for t in ["0.5", "0.75", "0.9"]:
                if t in e3:
                    lines.append(f"| 3D AP@{float(t):.2f} | {e3[t]:.4f} |")
            lines.append(f"")

        # Note
        note = model_cfg.get("note", "")
        if note:
            lines += [f"_{note}_", f""]

    # Per-folder prediction counts
    lines += [
        f"## Predicted Masks",
        f"",
        f"| Group | Folder | Masks written |",
        f"|-------|--------|---------------|",
    ]

    total = 0
    for group, date_folder, _ in leaf_dirs:
        out_dir = os.path.join(predictions_dir, model_name, group, date_folder)
        n = len([
            f for f in os.listdir(out_dir) if f.lower().endswith((".tif", ".tiff"))
        ]) if os.path.isdir(out_dir) else 0
        lines.append(f"| {group} | {date_folder} | {n} |")
        total += n

    lines += [f"", f"**Total: {total} masks**"]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"Summary written: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict nuclei masks on AKPS dataset with pretrained and finetuned models"
    )
    parser.add_argument("--config", default="configs/predict_akps_cluster_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and load existing model files")
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

    model_configs = cfg["models"]
    models_dir = cfg["output"]["models_dir"]
    predictions_dir = cfg["output"]["predictions_dir"]
    report_dir = os.path.join(predictions_dir, "_reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    use_bfloat16 = device.type == "cuda"

    # ------------------------------------------------------------------
    # 1. Discover all leaf image directories
    # ------------------------------------------------------------------
    if "predict_dirs" in cfg:
        leaf_dirs = collect_explicit_dirs(cfg["predict_dirs"])
    else:
        leaf_dirs = collect_leaf_dirs(cfg["data_root"])
    total_images = sum(
        len([f for f in os.listdir(p) if f.lower().endswith((".tif", ".tiff"))])
        for _, _, p in leaf_dirs
    )
    logging.info(f"Found {len(leaf_dirs)} image folders, {total_images} images total")
    for group, date_folder, _ in leaf_dirs:
        logging.info(f"  {group}/{date_folder}")

    # ------------------------------------------------------------------
    # 2. Load training data once (needed for finetuned models)
    # ------------------------------------------------------------------
    needs_training = any(not m.get("skip_training", False) for m in model_configs)

    train_images, train_masks, eval_images, eval_masks = [], [], [], []
    if needs_training and not args.skip_training:
        data_cfg = base_cfg["data"]
        training_cfg = base_cfg["training"]
        axis_map = {"Z": 0, "Y": 1, "X": 2}
        slice_axes = tuple(axis_map[a.upper()] for a in training_cfg.get("slice_axes", ["Z"]))

        logging.info("Collecting training pairs...")
        train_pairs = collect_image_mask_pairs(data_cfg["train_image_dirs"], data_cfg["gt_mapping"])
        eval_pairs = collect_image_mask_pairs(data_cfg["eval_image_dirs"], data_cfg["gt_mapping"])

        logging.info("Loading and slicing training data...")
        train_images, train_masks, _ = load_and_slice_pairs(
            train_pairs,
            min_masks_per_slice=training_cfg["min_masks_per_slice"],
            min_pixels=training_cfg["min_pixels"],
            slice_axes=slice_axes,
        )
        eval_images, eval_masks, _ = load_and_slice_pairs(
            eval_pairs,
            min_masks_per_slice=training_cfg["min_masks_per_slice"],
            min_pixels=training_cfg["min_pixels"],
            slice_axes=slice_axes,
        )
        logging.info(f"Training set: {len(train_images)} slices | Eval set: {len(eval_images)} slices")

    # ------------------------------------------------------------------
    # 3. Resolve model paths (train if needed)
    # ------------------------------------------------------------------
    resolved_models = []  # list of (model_cfg, model_path_or_base_model_name)

    for m in model_configs:
        if m.get("skip_training", False):
            # Use pretrained base model directly
            resolved_models.append((m, m["base_model"]))
            logging.info(f"Using pretrained model: {m['name']} ({m['base_model']})")
        else:
            expected_path = os.path.join(models_dir, "models", m["name"])
            if args.skip_training or os.path.exists(expected_path):
                if not os.path.exists(expected_path):
                    raise FileNotFoundError(
                        f"--skip-training set but model not found: {expected_path}"
                    )
                logging.info(f"Found existing model: {m['name']}")
                resolved_models.append((m, expected_path))
            else:
                model_path = train_model(
                    m, base_cfg, train_images, train_masks,
                    eval_images, eval_masks, models_dir, device,
                )
                resolved_models.append((m, model_path))

    # ------------------------------------------------------------------
    # 4. Predict with each model across all leaf directories
    # ------------------------------------------------------------------
    for m, model_path in resolved_models:
        model_name = m["name"]
        logging.info(f"\n{'='*60}")
        logging.info(f"Predicting with: {model_name}")
        logging.info(f"{'='*60}")

        model = models.CellposeModel(
            pretrained_model=model_path,
            device=device,
            use_bfloat16=use_bfloat16,
        )

        for group, date_folder, image_dir in leaf_dirs:
            out_dir = os.path.join(predictions_dir, model_name, group, date_folder)
            logging.info(f"  {group}/{date_folder}")
            predict_directory(model, image_dir, out_dir)

        write_summary(m, leaf_dirs, predictions_dir, report_dir)
        logging.info(f"Done: {model_name}")

    logging.info(f"\nAll predictions written to: {predictions_dir}")


if __name__ == "__main__":
    main()