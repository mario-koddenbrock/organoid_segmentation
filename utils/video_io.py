import logging

import cv2
from numpy import ndarray


def export_to_mp4(image: ndarray, video_file_path: str):
    first_frame = image[0]
    height, width = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (width, height))

    for frame in image:
        # Normalize and convert to 8-bit
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Convert grayscale to BGR for video codec
        frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    logging.info(f"Successfully created video: {video_file_path}")