import logging
import os

import pandas as pd

from utils.sam3model import SAM3Text

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from cellpose_adapt import io
from cellpose_adapt.metrics import calculate_segmentation_stats



def main():
    base_dir = "data/Organoids"
    organoid_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    all_metrics = []
    sam3 = SAM3Text(
        threshold = 0.5,
        mask_threshold = 0.5,
        text_prompt_option = 0,
    )

    for organoid_folder in organoid_folders:
        logging.info(f"Processing organoid: {organoid_folder}")
        organoid_path = os.path.join(base_dir, organoid_folder)

        image_folder = os.path.join(organoid_path, "images_cropped_isotropic")
        # membrane_folder = os.path.join(organoid_path, "labelmaps", "Membranes")
        nuclei_folder = os.path.join(organoid_path, "labelmaps", "Nuclei")

        if not all(os.path.exists(p) for p in [image_folder, nuclei_folder]):
            logging.warning(f"Skipping {organoid_folder}: missing one or more required subdirectories.")
            continue

        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.tif', '.tiff'))])
        # membrane_mask_files = sorted([os.path.join(membrane_folder, f) for f in os.listdir(membrane_folder) if f.endswith(('.tif', '.tiff'))])
        nuclei_mask_files = sorted([os.path.join(nuclei_folder, f) for f in os.listdir(nuclei_folder) if f.endswith(('.tif', '.tiff'))])


        if not all([image_files, nuclei_mask_files]):
            logging.warning(f"Skipping {organoid_folder}: missing one or more required files.")
            continue

        for image_file, nuclei_mask_file in zip(image_files, nuclei_mask_files):

            # Load the image and ground truth masks
            image_nuclei, gt_nuclei, image = io.load_image_with_gt(image_file, nuclei_mask_file, channel_to_segment=0)

            # this is only for the visualization, not effecting the segmentation
            # image[:,1,:,:] = rescale_intensity(image[:,1,:,:])

            # Initialize Model and Run Pipeline on Nuclei

            mask_nuclei = sam3.predict(image_nuclei)
            logging.info(f"Number of nuclei segments detected: {len(set(mask_nuclei.flat)) - 1}")

            metrics_nuclei = calculate_segmentation_stats(gt_nuclei, mask_nuclei, iou_threshold=0.75)
            logging.info(
                f"Performance (Nuclei): F1={metrics_nuclei.get('f1_score', 0):.3f}, Jaccard={metrics_nuclei.get('jaccard', 0):.3f}, ")


            # mask_membrane = sam3.predict(image_membrane)
            # logging.info(f"Number of membrane segments detected: {len(set(mask_membrane.flat)) - 1}")
            # metrics_membrane = calculate_segmentation_stats(gt_membrane, mask_membrane, iou_threshold=0.5)
            # logging.info(
            #     f"Performance (Membrane): F1={metrics_membrane.get('f1_score', 0):.3f}, Jaccard={metrics_membrane.get('jaccard', 0):.3f}, ")

            combined_metrics = {
                'image': os.path.basename(image_file),
                **{f'nuclei_{k}': v for k, v in metrics_nuclei.items()},
                # **{f'membrane_{k}': v for k, v in metrics_membrane.items()}
            }
            all_metrics.append(combined_metrics)
            break

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(results_dir, 'segmentation_metrics_sam3.csv'), index=False)
    logging.info(f"Metrics saved to {os.path.join(results_dir, 'segmentation_metrics_sam3.csv')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
