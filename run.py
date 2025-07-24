import argparse
import json
import logging
import os
import time

from skimage import segmentation
from skimage.filters import threshold_otsu

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import napari

from cellpose_adapt import core, caching
from cellpose_adapt import io
from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.utils import get_device


def main():
    args, cfg, device = init()

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

    # Create a foreground mask to apply watershed only on the organoid
    # This prevents watershed from "leaking" into the background
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
    

    show_napari(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_nuclei, metrics_membrane)


def init():
    parser = argparse.ArgumentParser(
        description="Run and visualize Cellpose results for a single image."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--membrane_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--nuclei_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the final pipeline configuration JSON file (e.g., best_cfg.json)."
    )
    parser.add_argument(
        "--cache_dir", type=str, required=False, help="Path to the cache directory for storing model outputs.",
    )
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"visualization_{timestamp}.log")
    if not os.path.exists(args.config_path):
        logging.error(f"Pipeline config file not found at {args.config_path}")
        raise (FileNotFoundError(f"Pipeline config file not found at {args.config_path}"))
    if not os.path.exists(args.nuclei_path):
        logging.error(f"Nuclei ground truth file not found at {args.nuclei_path}")
        raise (FileNotFoundError(f"Nuclei ground truth file not found at {args.nuclei_path}"))
    if not os.path.exists(args.image_path):
        logging.error(f"Image file not found at {args.image_path}")
        raise (FileNotFoundError(f"Image file not found at {args.image_path}"))
    if not os.path.exists(args.membrane_path):
        logging.error(f"Membrane image file not found at {args.membrane_path}")
        raise (FileNotFoundError(f"Membrane image file not found at {args.membrane_path}"))
    cfg = ModelConfig.from_json(args.config_path)
    device = get_device()
    return args, cfg, device


def show_napari(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_nuclei, metrics_membrane):

    viewer = napari.Viewer(title="Cellpose Single Image Visualization")
    is_3d = image.ndim == 4
    if is_3d:
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
        viewer.add_image(image[:, 0], name="Channel 1", colormap="cyan")
        viewer.add_image(image[:, 1], name="Channel 2", colormap="magenta")
    else:
        viewer.add_image(image, name="Image")

    if gt_nuclei is not None:
        viewer.add_labels(gt_nuclei, name="Nuclei (GT)", opacity=0.5)

    if gt_membrane is not None:
        viewer.add_labels(gt_membrane, name="Membrane (GT)", opacity=0.5)

    if mask_nuclei is not None:
        f1_score = metrics_nuclei.get('f1_score', 0.0)
        mask_name = f"Nuclei (Pred) (F1={f1_score:.2f})"
        viewer.add_labels(mask_nuclei, name=mask_name, opacity=0.7)

    if mask_membrane is not None:
        f1_score = metrics_membrane.get('f1_score', 0.0)
        mask_name = f"Membrane (Pred) (F1={f1_score:.2f})"
        viewer.add_labels(mask_membrane, name=mask_name, opacity=0.7)

    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()