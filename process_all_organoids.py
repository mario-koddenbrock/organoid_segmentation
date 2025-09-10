import os
import logging
import pandas as pd
from run import process_organoid
from utils import cli

def main():
    base_dir = "/Users/koddenbrock/Repository/organoid_segmentation/data/Organoids"
    organoid_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    args, cfg, device = cli.init()
    all_metrics = []

    for organoid_folder in organoid_folders:
        logging.info(f"Processing organoid: {organoid_folder}")
        organoid_path = os.path.join(base_dir, organoid_folder)

        image_folder = os.path.join(organoid_path, "images_cropped_isotropic")
        membrane_folder = os.path.join(organoid_path, "labelmaps", "Membranes")
        nuclei_folder = os.path.join(organoid_path, "labelmaps", "Nuclei")

        if not all(os.path.exists(p) for p in [image_folder, membrane_folder, nuclei_folder]):
            logging.warning(f"Skipping {organoid_folder}: missing one or more required subdirectories.")
            continue

        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.tif', '.tiff'))])
        membrane_mask_files = sorted([os.path.join(membrane_folder, f) for f in os.listdir(membrane_folder) if f.endswith(('.tif', '.tiff'))])
        nuclei_mask_files = sorted([os.path.join(nuclei_folder, f) for f in os.listdir(nuclei_folder) if f.endswith(('.tif', '.tiff'))])


        if not all([image_files, membrane_mask_files, nuclei_mask_files]):
            logging.warning(f"Skipping {organoid_folder}: missing one or more required files.")
            continue

        for image_file, membrane_mask_file, nuclei_mask_file in zip(image_files, membrane_mask_files, nuclei_mask_files):
            if "20240305_P013T_A006a" in image_file:
                logging.info("Skipping known problematic file.")
                continue

            metrics_nuclei, metrics_membrane = process_organoid(image_file, nuclei_mask_file, membrane_mask_file, cfg, device)

            combined_metrics = {
                'image': os.path.basename(image_file),
                **{f'nuclei_{k}': v for k, v in metrics_nuclei.items()},
                **{f'membrane_{k}': v for k, v in metrics_membrane.items()}
            }
            all_metrics.append(combined_metrics)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(results_dir, 'segmentation_metrics.csv'), index=False)
    logging.info(f"Metrics saved to {os.path.join(results_dir, 'segmentation_metrics.csv')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
