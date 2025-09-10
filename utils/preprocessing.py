import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms
from tqdm import tqdm

from utils.plotting import plot_slice


def rescale_intensity(image, q=(1, 99.65)):
    in_range = np.percentile(image, q)
    # plot_slice(image, "Before Rescaling Membrane Image (Intensity)")
    image_rescaled = exposure.rescale_intensity(image, in_range=(in_range[0], in_range[1]))
    # plot_slice(image_rescaled, "After Rescaling Membrane Image (Intensity)")
    return image_rescaled

def enhance_contrast(image):
    """
    Enhance the contrast of a 2D image using histogram equalization.
    Parameters:
    - image: np.ndarray
        2D array representing the image.
    Returns:
    - enhanced_image: np.ndarray
        Image with enhanced contrast.
    """
    # Apply histogram equalization
    enhanced_image = exposure.equalize_hist(image)
    return enhanced_image

def histogram_matching_by_99th_percentile(volume, multichannel=False):
    """
    Perform histogram matching on a 3D or 4D volume (with optional channels),
    using the slice with the highest 99th percentile intensity as reference.
    Parameters:
    - volume: np.ndarray
        Shape (Z, Y, X) for single-channel or (Z, C, Y, X) for multi-channel data.
    - multichannel: bool
        Whether the volume has multiple channels.
    Returns:
    - matched_volume: np.ndarray
        Volume after histogram matching.
    """
    if multichannel:
        #get the shape parameters in single variables
        Z, C, Y, X = volume.shape
        #create an empty array f the same shape as the input as container for the output
        matched_volume = np.empty_like(volume)
        #loop over the channels
        for c in range(C):
            print(f"Processing channel {c}")
            #stack/array of only one channel
            channel_volume = volume[:, c,...]
            #apply the function itself on the one channel array with multichannel false (by default) and save the result as a channel in the output array
            matched_volume[:, c,...] = histogram_matching_by_99th_percentile(channel_volume)
        return matched_volume
    else:
        #get the Z shape of the input array
        Z = volume.shape[0]
        #calculate 99th percentile intensity for every slice and save as 1D array
        percentiles = [np.percentile(volume[z], 99) for z in range(Z)]
        #find the index with highest 99th percentile intensity
        ref_index = np.argmax(percentiles)
        #define the slice with the correspondig index as reference
        reference_slice = volume[ref_index]
        #create a container for the result (array with same shape as input)
        matched_volume = np.empty_like(volume)
        #loop over slices in the stack and perform histogram matching function from skimage.exposure using the previously selected reference slice (and display a loading bar)
        for z in tqdm(range(Z), desc="Matching histograms"):
            matched_volume[z] = match_histograms(volume[z], reference_slice)
        return matched_volume