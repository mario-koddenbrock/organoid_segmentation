import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

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


def plot_segmentation_result(image, mask_nuclei, mask_membrane, gt_nuclei, gt_membrane, metrics_membrane, image_path, slice_index=None):
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
        metrics_membrane (dict): A dictionary of metrics for the membrane segmentation.
        image_path (str): The path to the image file.
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

    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    font_color = 'white'

    jaccard_score = metrics_membrane.get('jaccard', 0)
    f1_score = metrics_membrane.get('f1_score', 0)
    num_gt = len(set(gt_membrane.flat)) - 1
    num_pred = len(set(mask_membrane.flat)) - 1

    for title, data in slices.items():

        # Make figsize square for each subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        fig.suptitle(f'{title} - Jaccard: {jaccard_score:.3f} - F1@0.5: {f1_score:.3f}', fontsize=16, color=font_color)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.9) # Adjust spacing

        ax1, ax2 = axes.ravel()
        img_slice, m_nuc, m_mem, g_nuc, g_mem = data

        # --- Start of Bounding Box Calculation ---
        combined_gt_mask = (g_nuc > 0) | (g_mem > 0)
        rows, cols = np.where(combined_gt_mask)

        padding = 10
        if rows.size > 0 and cols.size > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # --- Make the bounding box square ---
            height = max_row - min_row
            width = max_col - min_col
            size = max(height, width)
            center_row = min_row + height // 2
            center_col = min_col + width // 2

            start_row = max(0, center_row - size // 2 - padding)
            end_row = min(combined_gt_mask.shape[0], center_row + size // 2 + padding)
            start_col = max(0, center_col - size // 2 - padding)
            end_col = min(combined_gt_mask.shape[1], center_col + size // 2 + padding)
        else:
            start_row, end_row = 0, combined_gt_mask.shape[0]
            start_col, end_col = 0, combined_gt_mask.shape[1]
        # --- End of Bounding Box Calculation ---

        # Normalize and create a color image
        nuclei_ch = img_slice[0].astype(np.float32)
        membrane_ch = img_slice[1].astype(np.float32)
        nuclei_ch /= (nuclei_ch.max() or 1.0)
        membrane_ch /= (membrane_ch.max() or 1.0)
        composite_img = np.stack([membrane_ch, nuclei_ch, nuclei_ch], axis=-1)

        # Plot Ground Truth
        ax1.imshow(composite_img)
        ax1.contour(g_nuc, colors='cyan', linewidths=0.5)
        ax1.contour(g_mem, colors='magenta', linewidths=0.5)
        ax1.set_title(f'Ground Truth ({num_gt})', color=font_color)
        ax1.axis('off')
        ax1.set_xlim(start_col, end_col)
        ax1.set_ylim(end_row, start_row)

        # Plot Predicted
        ax2.imshow(composite_img)
        ax2.contour(m_nuc, colors='cyan', linewidths=0.5)
        ax2.contour(m_mem, colors='magenta', linewidths=0.5)
        ax2.set_title(f'Prediction ({num_pred})', color=font_color)
        ax2.axis('off')
        ax2.set_xlim(start_col, end_col)
        ax2.set_ylim(end_row, start_row)

        # Create a legend inside the plot
        legend_elements = [Line2D([0], [0], color='cyan', lw=2, label='Nuclei'),
                           Line2D([0], [0], color='magenta', lw=2, label='Membrane')]
        ax1.legend(handles=legend_elements, loc='lower right', frameon=False, labelcolor='white')


        base_name = os.path.basename(image_path)
        file_name = os.path.splitext(base_name)[0]
        save_path = os.path.join(plots_dir, f'{file_name}_{title}.png')
        plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
