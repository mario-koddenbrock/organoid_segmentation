"""
Finetune CellposeSAM (cpsam) on organoid nuclei data.

Since cellpose training operates on 2D slices only, this script:
1. Loads all 3D organoid images and nuclei masks from data/Organoids/
2. Slices them along the Z-axis into individual 2D images
3. Filters out slices that have no masks (empty ground truth)
4. Splits into train/test sets (at the organoid level to avoid data leakage)
5. Finetunes the pretrained cpsam model
"""

import os
import logging
import argparse
import numpy as np
from tifffile import imread
from cellpose import models, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_image_mask_pairs(base_dir):
    """
    Walk data/Organoids/ and collect (image_path, nuclei_mask_path, organoid_folder) tuples.
    """
    pairs = []
    organoid_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])

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
                folder,
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


def slice_3d_to_2d(image_3d, mask_3d, min_masks=1):
    """
    Slice a 3D volume into 2D slices along the Z-axis.
    Filters out slices where the mask has fewer than `min_masks` unique labels
    (excluding background 0).

    Returns:
        images_2d: list of 2D numpy arrays
        masks_2d: list of 2D numpy arrays
        kept_indices: list of z-indices that were kept
    """
    images_2d = []
    masks_2d = []
    kept_indices = []

    for z in range(image_3d.shape[0]):
        mask_slice = mask_3d[z]
        n_labels = len(np.unique(mask_slice)) - (1 if 0 in mask_slice else 0)

        if n_labels < min_masks:
            continue

        images_2d.append(image_3d[z])
        masks_2d.append(mask_slice)
        kept_indices.append(z)

    return images_2d, masks_2d, kept_indices


def split_by_organoid(pairs, test_fraction=0.2, seed=42):
    """
    Split image/mask pairs into train and test sets at the organoid level
    to prevent data leakage between slices of the same volume appearing
    in both sets.

    Groups by organoid folder, then assigns entire organoids to train or test.
    """
    rng = np.random.default_rng(seed)

    # Group by organoid folder
    organoid_groups = {}
    for img_path, mask_path, folder in pairs:
        organoid_groups.setdefault(folder, []).append((img_path, mask_path))

    folders = sorted(organoid_groups.keys())
    rng.shuffle(folders)

    n_test = max(1, int(len(folders) * test_fraction))
    test_folders = set(folders[:n_test])
    train_folders = set(folders[n_test:])

    logging.info(f"Train organoids ({len(train_folders)}): {sorted(train_folders)}")
    logging.info(f"Test organoids  ({len(test_folders)}):  {sorted(test_folders)}")

    train_pairs = [(ip, mp) for ip, mp, f in pairs if f in train_folders]
    test_pairs = [(ip, mp) for ip, mp, f in pairs if f in test_folders]

    return train_pairs, test_pairs


def load_and_slice_pairs(pairs, min_masks_per_slice=1):
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
            nuclei_volume, mask_volume, min_masks=min_masks_per_slice
        )
        total_slices = nuclei_volume.shape[0]
        logging.info(
            f"  -> {len(kept)}/{total_slices} slices kept "
            f"(filtered {total_slices - len(kept)} empty slices)"
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
        "--min_train_masks", type=int, default=5,
        help="Cellpose min_train_masks: minimum masks for an image to be used in training"
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="Fraction of organoids to hold out for testing"
    )
    parser.add_argument(
        "--save_dir", type=str, default="models",
        help="Directory to save the finetuned model"
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


def main():
    args = parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = get_device()
    logging.info(f"Using device: {device}")

    # 1. Collect all image/mask pairs
    logging.info(f"Scanning {args.data_dir} for organoid data...")
    pairs = collect_image_mask_pairs(args.data_dir)
    logging.info(f"Found {len(pairs)} image/mask pairs across organoids")

    if not pairs:
        logging.error("No image/mask pairs found. Check your data directory.")
        return

    # 2. Split at organoid level
    train_pairs, test_pairs = split_by_organoid(
        pairs, test_fraction=args.test_fraction, seed=args.seed
    )

    # 3. Load and slice into 2D
    logging.info("Loading and slicing training data...")
    train_images, train_masks = load_and_slice_pairs(
        train_pairs, min_masks_per_slice=args.min_masks_per_slice
    )
    logging.info(f"Training set: {len(train_images)} slices")

    logging.info("Loading and slicing test data...")
    test_images, test_masks = load_and_slice_pairs(
        test_pairs, min_masks_per_slice=args.min_masks_per_slice
    )
    logging.info(f"Test set: {len(test_images)} slices")

    if not train_images:
        logging.error("No training slices after filtering. Lower --min_masks_per_slice.")
        return

    # 4. Initialize pretrained CellposeSAM model
    logging.info("Loading pretrained cpsam model...")
    model = models.CellposeModel(pretrained_model="cpsam", device=device)

    # 5. Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # 6. Finetune
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


if __name__ == "__main__":
    main()