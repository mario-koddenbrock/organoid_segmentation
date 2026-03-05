"""
Finetune CellposeSAM (cpsam) on organoid nuclei data.

Since cellpose training operates on 2D slices only, this script:
1. Loads organoid images and nuclei masks from the full image directory paths
   listed under data.train_image_dirs in the config file
2. Slices them along all three axes (Z, Y, X) into individual 2D images
3. Filters out slices that have no masks (empty ground truth)
4. Finetunes the pretrained cpsam model (train_seg handles augmentation internally)

Evaluation (before and after) is run on data.eval_image_dirs in both 2D
(slice-level) and 3D (full-volume, do_3D=True) modes.
All settings are read from a JSON config file (default:
configs/finetune_nuclei_config.json).
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from cellpose import metrics, models, train
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_finetune_config(path: str) -> dict:
    """Load and return the finetuning configuration from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_image_mask_pairs(image_dirs, gt_mapping):
    """
    Collect (image_path, mask_path) pairs from a list of full image directory paths.
    GT paths are derived from each image path using gt_mapping rules.
    Directories or individual images without a matching GT are skipped.
    """
    img_suffix = gt_mapping.get("img_suffix", ".tif")
    gt_suffix = gt_mapping.get("suffix", "_nuclei-labels.tif")
    replacements = gt_mapping.get("replace", [])

    pairs = []
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            logging.warning(f"Image directory not found, skipping: {image_dir}")
            continue

        # Derive the GT directory by applying path replacements to the image dir
        gt_dir = image_dir
        for old, new in replacements:
            gt_dir = gt_dir.replace(old, new)

        image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith((".tif", ".tiff"))
        ])

        dir_pairs = []
        for img_file in image_files:
            base_name = img_file
            if base_name.endswith(img_suffix):
                base_name = base_name[:-len(img_suffix)]
            gt_file = base_name + gt_suffix
            gt_path = os.path.join(gt_dir, gt_file)

            if not os.path.isfile(gt_path):
                logging.warning(f"No nuclei mask found for {img_file}, expected: {gt_path}")
                continue

            dir_pairs.append((os.path.join(image_dir, img_file), gt_path))

        logging.info(f"Found {len(dir_pairs)} pairs in: {image_dir}")
        pairs.extend(dir_pairs)

    return pairs


# ---------------------------------------------------------------------------
# Volume processing
# ---------------------------------------------------------------------------

def extract_nuclei_channel(volume):
    """
    Extract the nuclei channel (channel 0) from a 4D volume (Z, C, H, W).
    If the volume is already 3D (Z, H, W), return as-is.
    """
    if volume.ndim == 4:
        return volume[:, 0, :, :]
    return volume


def clean_mask(mask, min_pixels=64):
    """
    Remove mask instances smaller than `min_pixels` and relabel contiguously.
    Tiny instances cause cellpose flow computation to crash.
    """
    cleaned = np.zeros_like(mask)
    new_label = 1
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        region = mask == label_id
        if region.sum() < min_pixels:
            continue
        cleaned[region] = new_label
        new_label += 1
    return cleaned


def slice_3d_to_2d(image_3d, mask_3d, min_masks=1, min_pixels=64):
    """
    Slice a 3D volume into 2D slices along all three axes (Z, Y, X).
    Removes mask instances smaller than `min_pixels` pixels, then
    filters out slices where fewer than `min_masks` labels remain.
    """
    images_2d, masks_2d, kept_indices = [], [], []

    axis_names = ["Z", "Y", "X"]
    for axis in range(3):
        for i in range(image_3d.shape[axis]):
            img_slice = np.take(image_3d, i, axis=axis)
            msk_slice = np.take(mask_3d, i, axis=axis)

            mask_slice = clean_mask(msk_slice, min_pixels=min_pixels)
            if mask_slice.max() < min_masks:
                continue

            images_2d.append(img_slice.astype(np.float32))
            masks_2d.append(mask_slice.astype(np.int32))
            kept_indices.append((axis_names[axis], i))

    return images_2d, masks_2d, kept_indices


def load_and_slice_pairs(pairs, min_masks_per_slice=1, min_pixels=64):
    """
    Load 3D volumes from image/mask path pairs and slice into 2D.

    Returns:
        all_images:   list of 2D numpy arrays (grayscale nuclei channel)
        all_masks:    list of 2D numpy arrays (integer instance labels)
        all_volumes:  list of (nuclei_volume, mask_volume) 3D array tuples
    """
    all_images, all_masks, all_volumes = [], [], []

    for img_path, mask_path in pairs:
        logging.info(f"Loading {os.path.basename(img_path)}")
        volume = imread(img_path)
        mask_volume = imread(mask_path)

        nuclei_volume = extract_nuclei_channel(volume)

        if nuclei_volume.shape != mask_volume.shape:
            logging.warning(
                f"Shape mismatch: image {nuclei_volume.shape} vs mask {mask_volume.shape} "
                f"for {os.path.basename(img_path)}, skipping."
            )
            continue

        images_2d, masks_2d, kept = slice_3d_to_2d(
            nuclei_volume, mask_volume,
            min_masks=min_masks_per_slice,
            min_pixels=min_pixels,
        )
        total_slices = sum(nuclei_volume.shape[:3])
        logging.info(
            f"  -> {len(kept)}/{total_slices} slices kept across Z/Y/X axes "
            f"(filtered {total_slices - len(kept)} empty/small-mask slices)"
        )

        all_images.extend(images_2d)
        all_masks.extend(masks_2d)
        all_volumes.append((nuclei_volume, mask_volume))

    return all_images, all_masks, all_volumes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, images, masks, thresholds=(0.5, 0.75, 0.9)):
    """
    Run model on a list of 2D images and compute mean average precision.

    Returns:
        mean_ap: dict mapping threshold -> mean AP across images
        per_image_ap: np.ndarray of shape (n_images, n_thresholds)
    """
    n = len(images)
    logging.info(f"  Evaluating on {n} 2D slices...")
    pred_masks = []
    for i, img in enumerate(images):
        m, _, _ = model.eval(
            img, normalize=True, diameter=None, channels=None,
            min_size=15, flow_threshold=0.4, cellprob_threshold=0.0,
        )
        pred_masks.append(m)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logging.info(f"  Evaluated {i + 1}/{n} images")

    ap, tp, fp, fn = metrics.average_precision(masks, pred_masks, threshold=list(thresholds))
    mean_ap = {th: float(ap[:, i].mean()) for i, th in enumerate(thresholds)}
    return mean_ap, ap


def evaluate_model_3d(model, volumes, thresholds=(0.5, 0.75, 0.9)):
    """
    Run model on 3D volumes using do_3D=True (tri-plane inference) and compute
    mean average precision.

    Args:
        volumes: list of (nuclei_volume, mask_volume) tuples, each Z x H x W

    Returns:
        mean_ap: dict mapping threshold -> mean AP across volumes
        per_volume_ap: np.ndarray of shape (n_volumes, n_thresholds)
    """
    import numpy as np
    n = len(volumes)
    logging.info(f"  Evaluating on {n} 3D volumes...")
    pred_masks, gt_masks = [], []
    for i, (nuclei_vol, mask_vol) in enumerate(volumes):
        # Stack grayscale into 3-channel as expected by cpsam, channel axis last
        img = np.stack((nuclei_vol, np.zeros_like(nuclei_vol), np.zeros_like(nuclei_vol)), axis=-1)
        m, _, _ = model.eval(
            img, channel_axis=-1, z_axis=0,
            normalize=True, channels=None,
            do_3D=True, flow3D_smooth=2,
            diameter=30, bsize=256, batch_size=64,
            niter=1000, min_size=1000,
        )
        pred_masks.append(m)
        gt_masks.append(mask_vol)
        logging.info(f"  Evaluated volume {i + 1}/{n}")

    ap, tp, fp, fn = metrics.average_precision(gt_masks, pred_masks, threshold=list(thresholds))
    mean_ap = {th: float(ap[:, i].mean()) for i, th in enumerate(thresholds)}
    return mean_ap, ap


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(train_losses, test_losses, save_path):
    """Save a plot of train/test loss vs. epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    if len(test_losses) > 0:
        ax.plot(epochs, test_losses, label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Learning curve saved to {save_path}")


def plot_ap_comparison(ap_before, ap_after, save_path):
    """Save a grouped bar chart comparing AP before vs after finetuning."""
    thresholds = sorted(ap_before.keys())
    before_vals = [ap_before[t] for t in thresholds]
    after_vals = [ap_after[t] for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before finetuning")
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After finetuning")

    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Mean Average Precision")
    ax.set_title("Segmentation Performance: Before vs After Finetuning")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"AP comparison plot saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune CellposeSAM on organoid nuclei data"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/finetune_nuclei_config.json",
        help="Path to the finetuning configuration JSON file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda/mps/cpu). Auto-detected if not specified.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = get_device() if args.device is None else __import__("torch").device(args.device)
    logging.info(f"Using device: {device}")

    cfg = load_finetune_config(args.config)
    logging.info(f"Loaded config from: {args.config}")

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    train_image_dirs = data_cfg["train_image_dirs"]
    eval_image_dirs = data_cfg["eval_image_dirs"]
    gt_mapping = data_cfg["gt_mapping"]

    # 1. Collect image/mask pairs per split
    logging.info(f"Collecting training pairs from {len(train_image_dirs)} director(ies)...")
    train_pairs = collect_image_mask_pairs(train_image_dirs, gt_mapping)
    logging.info(f"Found {len(train_pairs)} training image/mask pairs")

    logging.info(f"Collecting eval pairs from {len(eval_image_dirs)} director(ies)...")
    eval_pairs = collect_image_mask_pairs(eval_image_dirs, gt_mapping)
    logging.info(f"Found {len(eval_pairs)} eval image/mask pairs")

    if not train_pairs:
        logging.error("No training pairs found. Check data_dir and train_folders in config.")
        return

    # 2. Load and slice into 2D (train_seg handles augmentation internally)
    logging.info("Loading and slicing training data...")
    train_images, train_masks, _ = load_and_slice_pairs(
        train_pairs,
        min_masks_per_slice=training_cfg["min_masks_per_slice"],
        min_pixels=training_cfg["min_pixels"],
    )
    logging.info(f"Training set: {len(train_images)} slices")

    logging.info("Loading and slicing eval data...")
    eval_images, eval_masks, eval_volumes = load_and_slice_pairs(
        eval_pairs,
        min_masks_per_slice=training_cfg["min_masks_per_slice"],
        min_pixels=training_cfg["min_pixels"],
    )
    logging.info(f"Eval set: {len(eval_images)} slices, {len(eval_volumes)} 3D volumes")

    if not train_images:
        logging.error("No training slices after filtering. Lower min_masks_per_slice in config.")
        return

    # 3. Initialize pretrained CellposeSAM model
    # use_bfloat16=False required: MPS does not support bfloat16 backward passes
    use_bfloat16 = device.type == "cuda"
    logging.info(f"Loading pretrained model '{model_cfg['base_model']}' (bfloat16={use_bfloat16})...")
    model = models.CellposeModel(
        pretrained_model=model_cfg["base_model"],
        device=device,
        use_bfloat16=use_bfloat16,
    )

    # 4. Evaluate pretrained model BEFORE finetuning (2D slices + 3D volumes)
    ap_before_2d = None
    ap_before_3d = None
    if eval_images:
        logging.info("Evaluating pretrained model on eval set (before finetuning) — 2D...")
        ap_before_2d, _ = evaluate_model(model, eval_images, eval_masks)
        for th, val in sorted(ap_before_2d.items()):
            logging.info(f"  2D AP@{th:.2f} = {val:.4f}")
    if eval_volumes:
        logging.info("Evaluating pretrained model on eval set (before finetuning) — 3D...")
        ap_before_3d, _ = evaluate_model_3d(model, eval_volumes)
        for th, val in sorted(ap_before_3d.items()):
            logging.info(f"  3D AP@{th:.2f} = {val:.4f}")

    # 5. Create output directory
    output_dir = os.path.join(model_cfg["save_dir"], "models")
    os.makedirs(output_dir, exist_ok=True)

    # 6. Finetune
    logging.info(
        f"Starting finetuning: {training_cfg['n_epochs']} epochs, "
        f"lr={training_cfg['learning_rate']}, batch_size={training_cfg['batch_size']}"
    )

    nimg_per_epoch = min(800, max(8, len(train_images)))
    logging.info(f"nimg_per_epoch={nimg_per_epoch}")

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=eval_images if eval_images else None,
        test_labels=eval_masks if eval_images else None,
        n_epochs=training_cfg["n_epochs"],
        learning_rate=training_cfg["learning_rate"],
        batch_size=training_cfg["batch_size"],
        weight_decay=training_cfg["weight_decay"],
        normalize=True,
        save_path=model_cfg["save_dir"],
        save_every=training_cfg["save_every"],
        save_each=False,
        min_train_masks=training_cfg["min_train_masks"],
        model_name=model_cfg["model_name"],
        scale_range=training_cfg["scale_range"],
        bsize=training_cfg["bsize"],
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=len(eval_images) if eval_images else None,
    )

    logging.info(f"Finetuned model saved to: {model_path}")
    logging.info(f"Final train loss: {train_losses[-1]:.4f}")

    # 7. Plot learning curves
    plot_path = os.path.join(output_dir, f"{model_cfg['model_name']}_learning_curves.png")
    plot_learning_curves(train_losses, test_losses, plot_path)

    # 8. Evaluate finetuned model AFTER finetuning (2D + 3D)
    finetuned_model = models.CellposeModel(
        pretrained_model=str(model_path), device=device, use_bfloat16=use_bfloat16,
    )

    ap_after_2d = None
    ap_after_3d = None
    if eval_images:
        logging.info("Evaluating finetuned model on eval set (after finetuning) — 2D...")
        ap_after_2d, _ = evaluate_model(finetuned_model, eval_images, eval_masks)
        for th, val in sorted(ap_after_2d.items()):
            logging.info(f"  2D AP@{th:.2f} = {val:.4f}")
    if eval_volumes:
        logging.info("Evaluating finetuned model on eval set (after finetuning) — 3D...")
        ap_after_3d, _ = evaluate_model_3d(finetuned_model, eval_volumes)
        for th, val in sorted(ap_after_3d.items()):
            logging.info(f"  3D AP@{th:.2f} = {val:.4f}")

    # 9. Print comparison summary
    logging.info("=" * 60)
    logging.info("Performance comparison (eval set):")
    for label, ap_before, ap_after in [
        ("2D", ap_before_2d, ap_after_2d),
        ("3D", ap_before_3d, ap_after_3d),
    ]:
        if ap_before is None or ap_after is None:
            continue
        logging.info(f"\n  [{label}] {'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}")
        logging.info(f"  [{label}] " + "-" * 44)
        for th in sorted(ap_before.keys()):
            before = ap_before[th]
            after = ap_after[th]
            delta = after - before
            sign = "+" if delta >= 0 else ""
            logging.info(f"  [{label}] AP@{th:<9.2f} {before:>10.4f} {after:>10.4f} {sign}{delta:>9.4f}")
    logging.info("=" * 60)

    # 10. Save AP comparison plots
    if ap_before_2d and ap_after_2d:
        ap_plot_path = os.path.join(output_dir, f"{model_cfg['model_name']}_ap_comparison_2d.png")
        plot_ap_comparison(ap_before_2d, ap_after_2d, ap_plot_path)
    if ap_before_3d and ap_after_3d:
        ap_plot_path = os.path.join(output_dir, f"{model_cfg['model_name']}_ap_comparison_3d.png")
        plot_ap_comparison(ap_before_3d, ap_after_3d, ap_plot_path)


if __name__ == "__main__":
    main()
