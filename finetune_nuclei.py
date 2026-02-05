"""
Finetune CellposeSAM (cpsam) on organoid nuclei data.

Since cellpose training operates on 2D slices only, this script:
1. Loads organoid images and nuclei masks from specified folders in data/Organoids/
2. Slices them along the Z-axis into individual 2D images
3. Filters out slices that have no masks (empty ground truth)
4. Randomly splits individual images into train/test sets
5. Finetunes the pretrained cpsam model

By default only 40x folders are included. Use --folders to specify exactly which
folders to use, or --all_folders to include everything (including 25x data).
"""

import os
import logging
import argparse
import numpy as np
from tifffile import imread
from cellpose import models, train, metrics
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_image_mask_pairs(base_dir, folders=None):
    """
    Walk data/Organoids/ and collect (image_path, nuclei_mask_path) tuples.
    If `folders` is given, only those subdirectories are included.
    """
    pairs = []
    all_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])

    if folders is not None:
        missing = set(folders) - set(all_folders)
        if missing:
            logging.warning(f"Folders not found in {base_dir}: {missing}")
        organoid_folders = [f for f in all_folders if f in folders]
    else:
        organoid_folders = all_folders

    logging.info(f"Using folders: {organoid_folders}")

    for folder in organoid_folders:
        image_dir = os.path.join(base_dir, folder, "images_cropped_isotropic")
        nuclei_dir = os.path.join(base_dir, folder, "labelmaps", "Nuclei")

        if not os.path.isdir(image_dir) or not os.path.isdir(nuclei_dir):
            logging.warning(f"Skipping {folder}: missing images or nuclei labelmaps.")
            continue

        image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith((".tif", ".tiff"))
        ])

        for img_file in image_files:
            # Derive nuclei mask filename: <name>.tif -> <name>_nuclei-labels.tif
            base_name = os.path.splitext(img_file)[0]
            mask_file = base_name + "_nuclei-labels.tif"
            mask_path = os.path.join(nuclei_dir, mask_file)

            if not os.path.isfile(mask_path):
                logging.warning(f"No nuclei mask found for {img_file}, expected {mask_file}")
                continue

            pairs.append((
                os.path.join(image_dir, img_file),
                mask_path,
            ))

    return pairs


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

    Returns:
        images_2d: list of 2D numpy arrays
        masks_2d: list of 2D numpy arrays
        kept_indices: list of (axis, index) tuples that were kept
    """
    images_2d = []
    masks_2d = []
    kept_indices = []

    axis_names = ["Z", "Y", "X"]
    for axis in range(3):
        for i in range(image_3d.shape[axis]):
            img_slice = np.take(image_3d, i, axis=axis)
            msk_slice = np.take(mask_3d, i, axis=axis)

            mask_slice = clean_mask(msk_slice, min_pixels=min_pixels)
            n_labels = mask_slice.max()  # contiguous labels from clean_mask

            if n_labels < min_masks:
                continue

            images_2d.append(img_slice.astype(np.float32))
            masks_2d.append(mask_slice.astype(np.int32))
            kept_indices.append((axis_names[axis], i))

    return images_2d, masks_2d, kept_indices


def augment_slices(images, masks, seed=None):
    """
    Apply basic augmentations to 2D image/mask pairs.

    For each input slice, generates augmented copies via:
      - horizontal flip
      - vertical flip
      - 90-degree rotation
      - 180-degree rotation
      - 270-degree rotation

    All transforms are applied identically to image and mask.
    Returns the original slices plus all augmented copies.
    """
    rng = np.random.default_rng(seed)
    aug_images = list(images)
    aug_masks = list(masks)

    for img, msk in zip(images, masks):
        # horizontal flip
        aug_images.append(np.flip(img, axis=1).copy())
        aug_masks.append(np.flip(msk, axis=1).copy())

        # vertical flip
        aug_images.append(np.flip(img, axis=0).copy())
        aug_masks.append(np.flip(msk, axis=0).copy())

        # 90, 180, 270 degree rotations
        for k in [1, 2, 3]:
            aug_images.append(np.rot90(img, k=k).copy())
            aug_masks.append(np.rot90(msk, k=k).copy())

    # shuffle so augmented copies are not grouped together
    order = rng.permutation(len(aug_images))
    aug_images = [aug_images[i] for i in order]
    aug_masks = [aug_masks[i] for i in order]

    return aug_images, aug_masks


def split_images(pairs, test_fraction=0.2, seed=42):
    """
    Randomly split image/mask pairs into train and test sets at the
    individual image level.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    n_test = max(1, int(len(pairs) * test_fraction))
    test_indices = set(indices[:n_test])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in test_indices]
    test_pairs = [pairs[i] for i in range(len(pairs)) if i in test_indices]

    logging.info(f"Train images ({len(train_pairs)}):")
    for ip, _ in train_pairs:
        logging.info(f"  {os.path.basename(ip)}")
    logging.info(f"Test images  ({len(test_pairs)}):")
    for ip, _ in test_pairs:
        logging.info(f"  {os.path.basename(ip)}")

    return train_pairs, test_pairs


def load_and_slice_pairs(pairs, min_masks_per_slice=1, min_pixels=64):
    """
    Load 3D volumes and slice into 2D, filtering empty slices.

    Returns:
        all_images: list of 2D numpy arrays (grayscale nuclei channel)
        all_masks: list of 2D numpy arrays (integer instance labels)
    """
    all_images = []
    all_masks = []

    for img_path, mask_path in pairs:
        logging.info(f"Loading {os.path.basename(img_path)}")
        volume = imread(img_path)
        mask_volume = imread(mask_path)

        # Extract nuclei channel from multichannel volume
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

    return all_images, all_masks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune CellposeSAM on organoid nuclei data"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/Organoids",
        help="Path to the Organoids data directory"
    )
    parser.add_argument(
        "--folders", type=str, nargs="+",
        default=[
            "20231108_P021N_40xSil_Hoechst_SiRActin",
            "20240220_P013T_40xSil_Hoechst_SiRActin",
            "20240305_P013T_40xSil_Hoechst_SiRActin",
            "20241009_P013T_40xSil_Hoechst_SiRActin",
        ],
        help="Organoid folders to include (default: 40x folders only)"
    )
    parser.add_argument(
        "--all_folders", action="store_true",
        help="Use all folders in data_dir, overrides --folders"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size (patches per GPU step)"
    )
    parser.add_argument(
        "--min_masks_per_slice", type=int, default=1,
        help="Minimum number of mask labels in a slice to keep it"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=64,
        help="Minimum pixels per mask instance; smaller instances are removed"
    )
    parser.add_argument(
        "--min_train_masks", type=int, default=5,
        help="Cellpose min_train_masks: minimum masks for an image to be used in training"
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="Fraction of images to hold out for testing"
    )
    parser.add_argument(
        "--save_dir", type=str, default=".",
        help="Base directory for model output (cellpose saves into <save_dir>/models/)"
    )
    parser.add_argument(
        "--model_name", type=str, default="cpsam_nuclei_finetuned",
        help="Name for the finetuned model"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split"
    )
    return parser.parse_args()


def evaluate_model(model, images, masks, thresholds=(0.5, 0.75, 0.9)):
    """
    Run model on a list of 2D images and compute average precision against
    ground truth masks at the given IoU thresholds.

    Returns:
        mean_ap: dict mapping threshold -> mean AP across images
        per_image_ap: np.ndarray of shape (n_images, n_thresholds)
    """
    n = len(images)
    logging.info(f"  Evaluating on {n} images...")
    pred_masks = []
    for i, img in enumerate(images):
        m, _, _ = model.eval(
            img, normalize=True, diameter=None,
            min_size=15, flow_threshold=0.4, cellprob_threshold=0.0,
        )
        pred_masks.append(m)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logging.info(f"  Evaluated {i + 1}/{n} images")

    ap, tp, fp, fn = metrics.average_precision(masks, pred_masks,
                                               threshold=list(thresholds))
    # ap shape: (n_images, n_thresholds)
    mean_ap = {th: float(ap[:, i].mean()) for i, th in enumerate(thresholds)}
    return mean_ap, ap


def plot_learning_curves(train_losses, test_losses, save_path):
    """Save a plot of train/test loss vs. epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    if test_losses:
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


def main():
    args = parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = get_device()
    logging.info(f"Using device: {device}")

    # 1. Collect image/mask pairs from specified folders
    folders = None if args.all_folders else args.folders
    logging.info(f"Scanning {args.data_dir} for organoid data...")
    pairs = collect_image_mask_pairs(args.data_dir, folders=folders)
    logging.info(f"Found {len(pairs)} image/mask pairs")

    if not pairs:
        logging.error("No image/mask pairs found. Check your data directory and --folders.")
        return

    # 2. Random train/test split at the image level
    train_pairs, test_pairs = split_images(
        pairs, test_fraction=args.test_fraction, seed=args.seed
    )

    # 3. Load and slice into 2D
    logging.info("Loading and slicing training data...")
    train_images, train_masks = load_and_slice_pairs(
        train_pairs, min_masks_per_slice=args.min_masks_per_slice,
        min_pixels=args.min_pixels,
    )
    logging.info(f"Training set: {len(train_images)} slices (before augmentation)")

    train_images, train_masks = augment_slices(train_images, train_masks)
    logging.info(f"Training set: {len(train_images)} slices (after augmentation)")

    logging.info("Loading and slicing test data...")
    test_images, test_masks = load_and_slice_pairs(
        test_pairs, min_masks_per_slice=args.min_masks_per_slice,
        min_pixels=args.min_pixels,
    )
    logging.info(f"Test set: {len(test_images)} slices")

    if not train_images:
        logging.error("No training slices after filtering. Lower --min_masks_per_slice.")
        return

    # 4. Initialize pretrained CellposeSAM model
    # use_bfloat16=False required: MPS does not support bfloat16 backward passes
    use_bfloat16 = device.type == "cuda"
    logging.info(f"Loading pretrained cpsam model (bfloat16={use_bfloat16})...")
    model = models.CellposeModel(pretrained_model="cpsam", device=device,
                                 use_bfloat16=use_bfloat16)

    # 5. Evaluate pretrained model on test set BEFORE finetuning
    ap_before = None
    if test_images:
        logging.info("Evaluating pretrained model on test set (before finetuning)...")
        ap_before, _ = evaluate_model(model, test_images, test_masks)
        for th, val in sorted(ap_before.items()):
            logging.info(f"  AP@{th:.2f} = {val:.4f}")

    # 6. Create save directory (cellpose adds models/ subdirectory internally)
    output_dir = os.path.join(args.save_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    # 7. Finetune
    logging.info(
        f"Starting finetuning: {args.n_epochs} epochs, "
        f"lr={args.learning_rate}, batch_size={args.batch_size}"
    )

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_images else None,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=0.1,
        normalize=True,
        save_path=args.save_dir,
        save_every=50,
        save_each=False,
        min_train_masks=args.min_train_masks,
        model_name=args.model_name,
        scale_range=0.5,
        bsize=256,
    )

    logging.info(f"Finetuned model saved to: {model_path}")
    logging.info(f"Final train loss: {train_losses[-1]:.4f}")
    if test_losses:
        logging.info(f"Final test loss:  {test_losses[-1]:.4f}")

    # 8. Plot learning curves
    plot_path = os.path.join(output_dir, f"{args.model_name}_learning_curves.png")
    plot_learning_curves(train_losses, test_losses, plot_path)

    # 9. Evaluate finetuned model on test set AFTER finetuning
    if test_images:
        logging.info("Evaluating finetuned model on test set (after finetuning)...")
        finetuned_model = models.CellposeModel(
            pretrained_model=model_path, device=device, use_bfloat16=use_bfloat16,
        )
        ap_after, _ = evaluate_model(finetuned_model, test_images, test_masks)
        for th, val in sorted(ap_after.items()):
            logging.info(f"  AP@{th:.2f} = {val:.4f}")

        # 10. Print comparison summary
        logging.info("=" * 50)
        logging.info("Performance comparison (test set):")
        logging.info(f"{'Metric':<12} {'Before':>10} {'After':>10} {'Delta':>10}")
        logging.info("-" * 44)
        for th in sorted(ap_before.keys()):
            before = ap_before[th]
            after = ap_after[th]
            delta = after - before
            sign = "+" if delta >= 0 else ""
            logging.info(f"AP@{th:<9.2f} {before:>10.4f} {after:>10.4f} {sign}{delta:>9.4f}")
        logging.info("=" * 50)

        # 11. Save AP comparison plot
        ap_plot_path = os.path.join(
            output_dir, f"{args.model_name}_ap_comparison.png"
        )
        plot_ap_comparison(ap_before, ap_after, ap_plot_path)


if __name__ == "__main__":
    main()