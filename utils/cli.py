import argparse
import logging
import os
import time

from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.logger import setup_logging
from cellpose_adapt.utils import get_device

logger = logging.getLogger(__name__)


def init():
    parser = argparse.ArgumentParser(
        description="Run and visualize Cellpose results for a single image."
    )
    parser.add_argument(
        "--image_path", type=str, required=False, help="Path to the input image."
    )
    parser.add_argument(
        "--membrane_path", type=str, required=False, help="Path to the input image."
    )
    parser.add_argument(
        "--nuclei_path", type=str, required=False, help="Path to the input image."
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
    if args.nuclei_path and not os.path.exists(args.nuclei_path):
        logging.warning(f"Nuclei ground truth file not found at {args.nuclei_path}")
        raise FileNotFoundError(f"Nuclei ground truth file not found at {args.nuclei_path}")
    if args.image_path and not os.path.exists(args.image_path):
        logging.error(f"Image file not found at {args.image_path}")
        raise FileNotFoundError(f"Image file not found at {args.image_path}")
    if args.membrane_path and not os.path.exists(args.membrane_path):
        logging.error(f"Membrane image file not found at {args.membrane_path}")
        raise FileNotFoundError(f"Membrane image file not found at {args.membrane_path}")
    cfg = ModelConfig.from_json(args.config_path)
    device = get_device()
    return args, cfg, device
