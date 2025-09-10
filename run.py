import logging
import os

from cellpose_adapt.plotting.napari_utils import show_napari

from utils.preprocessing import rescale_intensity

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from utils.plotting import plot_segmentation_result

from skimage import segmentation
from utils import cli
from cellpose_adapt import core
from cellpose_adapt import io
from cellpose_adapt.metrics import calculate_segmentation_stats


def process_organoid(image_path, nuclei_path, membrane_path, cfg, device):
    """
    Processes a single organoid.
    """
    # Load the image and ground truth masks
    image_nuclei, gt_nuclei, image = io.load_image_with_gt(image_path, nuclei_path, channel_to_segment=0)
    image_membrane, gt_membrane, _ = io.load_image_with_gt(image_path, membrane_path, channel_to_segment=1)

    # this is only for the visualization, not effecting the segmentation
    # image[:,1,:,:] = rescale_intensity(image[:,1,:,:])

    # Initialize Model and Run Pipeline on Nuclei
    model = core.initialize_model(cfg.model_name, device=device)
    cache_dir = ".cache"
    runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)
    # mask_nuclei, _ = runner.run(image_nuclei)
    mask_nuclei = gt_nuclei
    logging.info(f"Number of nuclei segments detected: {len(set(mask_nuclei.flat)) - 1}")

    metrics_nuclei = calculate_segmentation_stats(gt_nuclei, mask_nuclei, iou_threshold=0.75)
    logging.info(
        f"Performance (Nuclei): F1={metrics_nuclei.get('f1_score', 0):.3f}, Jaccard={metrics_nuclei.get('jaccard', 0):.3f}, ")

    # Enhance the membrane image for better segmentation
    image_membrane_rescaled = rescale_intensity(image_membrane, q=(0, 99.2))
    # image_membrane_rescaled = histogram_matching_by_99th_percentile(image_membrane, False)

    # Apply watershed
    mask_membrane = segmentation.watershed(
        image=image_membrane_rescaled,
        markers=mask_nuclei,
    )
    # plot_slice(mask_membrane, "Watershed Segmentation with Mask")

    # Post-process the membrane mask
    mask_membrane[mask_membrane == 1] = 0
    # plot_slice(mask_membrane, "Watershed Segmentation with Mask (Background Set to 0)")

    # Evaluate
    metrics_membrane = calculate_segmentation_stats(gt_membrane, mask_membrane, iou_threshold=0.5)

    logging.info(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")
    logging.info(
        f"Performance (Membrane): F1={metrics_membrane.get('f1_score', 0):.3f}, Jaccard={metrics_membrane.get('jaccard', 0):.3f}, ")

    plot_segmentation_result(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_membrane, image_path)
    # show_napari(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_nuclei, metrics_membrane)
    return metrics_nuclei, metrics_membrane


def main():
    args, cfg, device = cli.init()
    process_organoid(args.image_path, args.nuclei_path, args.membrane_path, cfg, device)


if __name__ == "__main__":
    main()