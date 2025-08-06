import numpy as np
from skimage import exposure

from utils.plotting import plot_slice


def rescale_intensity(image, q=(1, 99.65)):
    in_range = np.percentile(image, q)
    # plot_slice(image, "Before Rescaling Membrane Image (Intensity)")
    image_rescaled = exposure.rescale_intensity(image, in_range=(in_range[0], in_range[1]))
    # plot_slice(image_rescaled, "After Rescaling Membrane Image (Intensity)")
    return image_rescaled