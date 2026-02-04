import logging

import numpy as np
from cachetools import cached, LRUCache
from sam3.model_builder import build_sam3_video_predictor

from utils.base_model import BaseModel
from utils.video_io import export_to_mp4

# transformers provides Sam3Processor/Sam3Model in newer versions; guard import to keep static checks quiet
# try:
#     from transformers import Sam3Processor, Sam3Model, pipeline
# except Exception as _e:  # pragma: no cover
#     Sam3Processor = None
#     Sam3Model = None
#     _import_err = _e

# Module logger
logger = logging.getLogger(__name__)

# Cache for the model and processor
# We can use a simple LRUCache here. maxsize=2 is a reasonable default if you expect
# to work with models on both CPU and GPU.
model_cache = LRUCache(maxsize=2)


@cached(model_cache)
def _load_sam3_model():
    return build_sam3_video_predictor()



class SAM3Text(BaseModel):

    # Predefined text prompts for nuclei segmentation
    TEXT_PROMPTS = {
        0: "blob",
        1: "cell nucleus",
        2: "nuclei",
        3: "round nucleus",
        4: "spherical nucleus",
        5: "elongated nucleus",
        6: "irregularly shaped nucleus",
        7: "bright nucleus",
        8: "dark nucleus",
        9: "fluorescent nucleus",
        10: "stained nucleus",
        11: "DNA-rich region",
        12: "chromatin",
        13: "dense chromatin",
        14: "cellular nucleus",
        15: "single nucleus",
        16: "cluster of nuclei",
        17: "isolated nucleus",
        18: "overlapping nuclei",
        19: "round blob",
        20: "small round object",
        21: "3D spherical object",
        22: "cellular blob",
        23: "nuclear region",
        24: "fluorescent blob",
        25: "bright round structure",
        26: "nuclear body",
        27: "cellular DNA region",
        28: "granular nucleus",
        29: "dense blob",
        30: "elongated blob",
        31: "spherical structure",
        32: "ovoid nucleus",
        33: "nucleus in 3D",
        34: "nuclear volume",
        35: "microscopy nucleus",
        36: "fluorescent sphere",
        37: "cell interior blob",
        38: "round dense body",
        39: "isolated 3D nucleus",
        40: "nuclear cluster",
        41: "overlapping blobs",
        42: "stained DNA region",
        43: "bright granular structure",
        44: "cellular sphere",
        45: "microscopy blob",
        46: "nucleus",
    }

    def __init__(
        self,
        model_name: str = "SAM3Text",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        text_prompt_option: int = 0,
        video_path: str = ".temp/video_input",
        **kwargs,
    ):
        self._video_predictor = None
        logger.debug(
            "Initializing SAM3 model wrapper: model_name=%s, threshold=%s, mask_threshold=%s, text_prompt_option=%s, kwargs=%s",
            model_name,
            threshold,
            mask_threshold,
            text_prompt_option,
            {k: v for k, v in kwargs.items()},
        )
        super().__init__(model_name, **kwargs)

        self.text_prompt_option = text_prompt_option
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None
        self._video_path = video_path

    def load_model(self):
        """Load the model for text-prompted segmentation."""
        logger.debug("Called load_model()")
        if self._model is not None:
            logger.debug("Model already loaded; skipping load_model")
            return

        try:
            # This will use the cached version if available for the current device
            self._video_predictor = _load_sam3_model()
        except Exception as e:
            logger.exception("Failed to load SAM3 model: %s", e)
            raise


    def predict(self, image: np.ndarray) -> np.ndarray:
        """Text-prompted segmentation using SAM3 image model."""
        logger.debug(
            "predict() called with image shape=%s, dtype=%s",
            getattr(image, "shape", None),
            getattr(image, "dtype", None),
        )
        if image.ndim not in (2, 3):
            logger.error("Invalid image dimensionality: %s", image.shape)
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        if self._video_predictor is None:
            logger.debug("Model not loaded in predict(); calling load_model()")
            self.load_model()

        assert self._video_predictor is not None


        # Prepare the video input
        export_to_mp4(image, self._video_path)

        # Get the text prompt based on the selected option
        text_prompt = self.TEXT_PROMPTS.get(self.text_prompt_option, self.TEXT_PROMPTS[0])
        logger.debug("Using text prompt option %s: '%s'", self.text_prompt_option, text_prompt)

        # Start a session
        response = self._video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self._video_path,
            )
        )


        labeled_masks = np.zeros_like(image, dtype=np.uint16)
        for frame_idx in range(image.shape[0] if image.ndim == 3 else 1):
            logger.debug("Processing frame %d", frame_idx)

            response = self._video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=response["session_id"],
                    frame_index=0,
                    text=text_prompt,
                )
            )
            labeled_masks[frame_idx] = response["outputs"]


        return labeled_masks