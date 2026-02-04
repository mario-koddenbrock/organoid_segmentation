import logging
import os

from utils.video_io import export_to_mp4

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from cellpose_adapt import io


def main():
    base_dir = "data/Organoids/"
    output_dir = "data/Organoid_Videos/"
    os.makedirs(output_dir, exist_ok=True)
    organoid_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    for organoid_folder in organoid_folders:
        logging.info(f"Processing organoid: {organoid_folder}")
        organoid_path = os.path.join(base_dir, organoid_folder)

        video_path = os.path.join(base_dir, organoid_folder, "videos")
        os.makedirs(video_path, exist_ok=True)

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
            image_nuclei, _, _ = io.load_image_with_gt(image_file, nuclei_mask_file, channel_to_segment=0)

            video_file_name = os.path.splitext(os.path.basename(image_file))[0] + ".mp4"
            video_file_path = os.path.join(output_dir, video_file_name)

            if len(image_nuclei) == 0:
                logging.warning(f"No frames found for {image_file}")
                continue

            export_to_mp4(image_nuclei, video_file_path)





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
