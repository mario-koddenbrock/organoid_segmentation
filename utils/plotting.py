import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_slice(image, title="", slice_index=None):
    """
    Plots a single slice of the image for a given channel.

    Args:
        image (np.ndarray): The input image, expected to have shape (Z, H, W) or (Z, C, H, W).
        slice_index (int): The index of the slice to plot.
        channel (int): The channel to plot (default is 0).
    """

    if slice_index is None:
        # Use the middle slice if no index is provided
        if image.ndim == 3:
            slice_index = image.shape[0] // 2

    img_slice = image[slice_index, ...]

    title_str = f"{title} - Slice {slice_index}" if title else f"Slice {slice_index}"

    plt.figure(figsize=(8, 8))
    plt.imshow(img_slice, cmap='viridis')
    plt.title(title_str)
    plt.axis('off')
    plt.show()


def plot_segmentation_result(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, slice_index=None):
    """
    Plots a comparison of predicted vs. ground truth segmentation for the middle
    slice in each spatial dimension (Z, Y, X) on separate figures.

    Args:
        image (np.ndarray): The input image, expected to have 2 channels (nuclei, membrane).
                            Can be 3D (C, H, W) or 4D (Z, C, H, W).
        mask_nuclei (np.ndarray): The predicted nuclei segmentation mask.
        mask_membrane (np.ndarray): The predicted membrane segmentation mask.
        gt_nuclei (np.ndarray): The ground truth nuclei segmentation mask.
        gt_membrane (np.ndarray): The ground truth membrane segmentation mask.
        slice_index (any, optional): This argument is ignored. The function always
                                     uses the middle slice for each dimension.
    """
    if image.ndim not in [3, 4]:
        raise ValueError("Image must be 3D (C, H, W) or 4D (Z, C, H, W)")

    if image.ndim == 3:  # If image is 2D (C, H, W), add a dummy Z dimension
        image = image[np.newaxis, ...]
        mask_nuclei = mask_nuclei[np.newaxis, ...]
        mask_membrane = mask_membrane[np.newaxis, ...]
        gt_nuclei = gt_nuclei[np.newaxis, ...]
        gt_membrane = gt_membrane[np.newaxis, ...]

    z, c, h, w = image.shape
    mid_z, mid_y, mid_x = z // 2, h // 2, w // 2

    # Slices for Z, Y, X dimensions
    slices = {
        'Z-Slice': (
            image[mid_z, ...],
            mask_nuclei[mid_z, ...], mask_membrane[mid_z, ...],
            gt_nuclei[mid_z, ...], gt_membrane[mid_z, ...]
        ),
        'Y-Slice': (
            image[:, :, mid_y, :].transpose(1, 0, 2),  # C, Z, W
            mask_nuclei[:, mid_y, :],  # Z, W
            mask_membrane[:, mid_y, :],
            gt_nuclei[:, mid_y, :],
            gt_membrane[:, mid_y, :]
        ),
        'X-Slice': (
            image[:, :, :, mid_x].transpose(1, 0, 2),  # C, Z, H
            mask_nuclei[:, :, mid_x],  # Z, H
            mask_membrane[:, :, mid_x],
            gt_nuclei[:, :, mid_x],
            gt_membrane[:, :, mid_x]
        )
    }

    for title, data in slices.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        fig.suptitle(f'Segmentation Result - {title}', fontsize=16)

        ax1, ax2 = axes.ravel()
        img_slice, m_nuc, m_mem, g_nuc, g_mem = data

        # Normalize and create a color image
        nuclei_ch = img_slice[0].astype(np.float32)
        membrane_ch = img_slice[1].astype(np.float32)
        nuclei_ch /= (nuclei_ch.max() or 1.0)
        membrane_ch /= (membrane_ch.max() or 1.0)
        composite_img = np.stack([membrane_ch, nuclei_ch, nuclei_ch], axis=-1)

        # Plot Predicted
        ax1.imshow(composite_img)
        ax1.contour(m_nuc, colors='cyan', linewidths=0.5)
        ax1.contour(m_mem, colors='magenta', linewidths=0.5)
        ax1.set_title('Prediction')
        ax1.axis('off')

        # Plot Ground Truth
        ax2.imshow(composite_img)
        ax2.contour(g_nuc, colors='cyan', linewidths=0.5)
        ax2.contour(g_mem, colors='magenta', linewidths=0.5)
        ax2.set_title('Ground Truth')
        ax2.axis('off')

        # Add a legend to the figure
        legend_elements = [Line2D([0], [0], color='cyan', lw=2, label='Nuclei Mask'),
                           Line2D([0], [0], color='magenta', lw=2, label='Membrane Mask')]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        break

    plt.show()