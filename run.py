import logging
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from skimage import segmentation
from skimage.filters import threshold_otsu

from utils import cli, plotting
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
    image_membrane, gt_membrane, _ = io.load_image_with_gt(args.image_path, args.nuclei_path, channel_to_segment=1)
    if image_membrane is None:
        logging.error(f"Failed to load image from {args.image_path}")
        return

    # Initialize Model and Run Pipeline on Nuclei
    model = core.initialize_model(cfg.model_name, device=device)
    cache_dir = args.cache_dir if args.cache_dir else ".cache"
    runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)
    mask_nuclei, duration = runner.run(image_nuclei)

    # Create a foreground mask to apply watershed on the nuclei mask
    thresh = threshold_otsu(image_membrane)
    foreground_mask = image_membrane > thresh

    # Ensure all nuclei are included in the foreground mask for watershed seeding
    # by taking the union of the thresholded membrane and the nuclei mask.
    foreground_mask = foreground_mask | (mask_nuclei > 0)

    # Watershed the membrane image with nuclei mask as seeds
    mask_membrane = segmentation.watershed(image_membrane, mask_nuclei, mask=foreground_mask)
    logging.info(f"Number of nuclei segments detected: {len(set(mask_nuclei.flat)) - 1}")
    logging.info(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")

    # --- Evaluate and Launch Napari Viewer ---
    metrics_nuclei = calculate_segmentation_stats(gt_nuclei, mask_nuclei)
    logging.info(f"Performance (Nuclei): "
                 f"F1={metrics_nuclei.get('f1_score', 0):.3f}, "
                 f"P={metrics_nuclei.get('precision', 0):.3f}, "
                 f"R={metrics_nuclei.get('recall', 0):.3f}")
    
    metrics_membrane = calculate_segmentation_stats(gt_membrane, mask_membrane)
    logging.info(f"Performance (Membrane): "
                 f"F1={metrics_membrane.get('f1_score', 0):.3f}, "
                 f"P={metrics_membrane.get('precision', 0):.3f}, "
                 f"R={metrics_membrane.get('recall', 0):.3f}")

    plotting.show_napari(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_nuclei, metrics_membrane)


if __name__ == "__main__":
    main()