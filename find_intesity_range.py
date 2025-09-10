import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.preprocessing import rescale_intensity

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from skimage import segmentation
from utils import cli
from cellpose_adapt import io
from cellpose_adapt.metrics import calculate_segmentation_stats


def main():
    args, cfg, device = cli.init()

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Load the image and ground truth masks
    image_nuclei, gt_nuclei, image = io.load_image_with_gt(args.image_path, args.nuclei_path, channel_to_segment=0)
    image_membrane, gt_membrane, _ = io.load_image_with_gt(args.image_path, args.membrane_path, channel_to_segment=1)

    # Initialize Model and Run Pipeline on Nuclei
    # model = core.initialize_model(cfg.model_name, device=device)
    # cache_dir = args.cache_dir if args.cache_dir else ".cache"
    # runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)
    # mask_nuclei, _ = runner.run(image_nuclei)
    mask_nuclei = gt_nuclei

    q_max_values = np.linspace(97, 100, 20)
    iou = []
    f1 = []
    for q_max in tqdm(q_max_values, desc="Processing Q Max Values", unit="im"):

        # Enhance the membrane image for better segmentation
        image_membrane_rescaled = rescale_intensity(image_membrane, q=(1, q_max))

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

        logging.debug(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")
        logging.debug(f"Performance (Membrane): F1={metrics_membrane.get('f1_score', 0):.3f}, Jaccard={metrics_membrane.get('jaccard', 0):.3f}, ")

        iou.append(metrics_membrane.get('jaccard', 0))
        f1.append(metrics_membrane.get('f1_score', 0))

    optimal_q_max = q_max_values[np.argmax(iou)]
    logging.info(f"Optimal Q Max: {optimal_q_max:.2f} with IoU={max(iou):.3f} and F1={max(f1):.3f}")

    plt.plot(q_max_values, iou, label='IoU', marker='o')
    plt.plot(q_max_values, f1, label='F1@IoU=0.5', marker='x')
    plt.axvline(optimal_q_max, color='red', linestyle='--', label=f'Optimal Q Max: {optimal_q_max:.2f}')
    plt.legend()
    plt.xlabel("Max Intensity Percentile")
    plt.ylabel("Score")
    plt.grid()
    plt.title("Effect of Max Intensity Percentile on Membrane Segmentation")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "membrane_segmentation_q_max_analysis.png"))
    plt.show()


    q_min_values = np.linspace(0, 95, 20)
    iou = []
    f1 = []
    for q_min in tqdm(q_min_values, desc="Processing Q Min Values", unit="im"):

        # Enhance the membrane image for better segmentation
        image_membrane_rescaled = rescale_intensity(image_membrane, q=(1, q_min))

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

        logging.debug(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")
        logging.debug(f"Performance (Membrane): F1={metrics_membrane.get('f1_score', 0):.3f}, Jaccard={metrics_membrane.get('jaccard', 0):.3f}, ")

        iou.append(metrics_membrane.get('jaccard', 0))
        f1.append(metrics_membrane.get('f1_score', 0))

    plt.plot(q_min_values, iou, label='IoU', marker='o')
    plt.plot(q_min_values, f1, label='F1@IoU=0.5', marker='x')
    plt.legend()
    plt.xlabel("Min Intensity Percentile")
    plt.ylabel("Score")
    plt.grid()
    plt.title("Effect of Min Intensity Percentile on Membrane Segmentation")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "membrane_segmentation_q_min_analysis.png"))
    plt.show()





if __name__ == "__main__":
    main()