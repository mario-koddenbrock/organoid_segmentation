import logging
import os

import numpy as np
from cellpose_adapt.plotting.napari_utils import show_napari

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from utils.plotting import plot_segmentation_result, plot_slice

from skimage import filters, segmentation
from scipy import ndimage as ndi
from utils import cli
from cellpose_adapt import core
from cellpose_adapt import io
from cellpose_adapt.metrics import calculate_segmentation_stats


def main():
    args, cfg, device = cli.init()

    # Load the image and ground truth nuclei mask
    image_nuclei, gt_nuclei, image = io.load_image_with_gt(args.image_path, args.nuclei_path, channel_to_segment=0)
    if image_nuclei is None:
        logging.error(f"Failed to load image from {args.image_path}")
        return

    # Load the image and ground truth membrane mask
    image_membrane, gt_membrane, _ = io.load_image_with_gt(args.image_path, args.membrane_path, channel_to_segment=1)


    # Initialize Model and Run Pipeline on Nuclei
    model = core.initialize_model(cfg.model_name, device=device)
    cache_dir = args.cache_dir if args.cache_dir else ".cache"
    runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)
    # mask_nuclei, _ = runner.run(image_nuclei)
    mask_nuclei = gt_nuclei
    logging.info(f"Number of nuclei segments detected: {len(set(mask_nuclei.flat)) - 1}")

    metrics_nuclei = calculate_segmentation_stats(gt_nuclei, mask_nuclei, iou_threshold=0.75)
    logging.info(f"Performance (Nuclei): F1={metrics_nuclei.get('f1_score', 0):.3f}, Jaccard={metrics_nuclei.get('jaccard', 0):.3f}, ")

    gradient = ndi.generic_gradient_magnitude(image_membrane.astype(float), ndi.sobel)
    membrane_super_smooth = filters.gaussian(image_membrane, sigma=20)
    # plot_slice(membrane_super_smooth, "Smoothed Membrane Image")

    foreground_mask = membrane_super_smooth > filters.threshold_otsu(membrane_super_smooth)
    # plot_slice(foreground_mask, "Foreground Mask with Otsu Threshold")

    # Step 3: Apply watershed
    mask_membrane = segmentation.watershed(
        image=gradient,
        markers=mask_nuclei,
        mask=foreground_mask,
    )
    # plot_slice(mask_membrane, "Watershed Segmentation with Mask")
    mask_membrane[mask_membrane == 1] = 0
    # plot_slice(mask_membrane, "Watershed Segmentation with Mask (Background Set to 0)")

    # Watershed the membrane image with nuclei mask as seeds
    # mask_membrane = segmentation.watershed(image_membrane, mask_nuclei, mask=foreground_mask)
    logging.info(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")

    # --- Evaluate and Launch Napari Viewer ---
    metrics_membrane = calculate_segmentation_stats(gt_membrane, mask_membrane, iou_threshold=0.5)
    logging.info(f"Performance (Membrane): F1={metrics_membrane.get('f1_score', 0):.3f}, Jaccard={metrics_membrane.get('jaccard', 0):.3f}, ")


    plot_segmentation_result(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane)
    show_napari(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_nuclei, metrics_membrane)


if __name__ == "__main__":
    main()