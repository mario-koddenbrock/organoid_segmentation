"""Compute per-nucleus morphology features directly from label masks.

Used for the manual-vs-auto segmentation cross-domain comparison, where both
domains' features must be computed with the IDENTICAL method to be a fair test —
unlike Joshua's already-computed CSVs (used elsewhere in `analysis/data.py`),
which we can't use for the auto side here without also recomputing the manual
side the same way.

Feature set is a deliberately simpler/self-contained re-derivation of the same
kind of features as the main CSVs (volume, surface area, sphericity, solidity,
ellipsoid axes, aspect ratios, prolate/oblate shape ratios, positional context) —
not guaranteed to numerically match Joshua's exact formulas, but internally
consistent between manual and auto masks, which is what the comparison needs.
"""

import os
import re

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops, marching_cubes, mesh_surface_area

MIN_VOXELS = 30  # drop tiny fragments (segmentation noise)


def spacing_for(img_name: str) -> float:
    return 0.26 if "20241023" in img_name else 0.324


def extract_features_from_mask(mask_path: str) -> pd.DataFrame:
    """One row per nucleus in this label mask."""
    fname = os.path.basename(mask_path)
    img_name = re.sub(r"_cropped_isotropic.*", "", fname)
    spacing_um = spacing_for(img_name)
    spacing = (spacing_um, spacing_um, spacing_um)

    lm = tifffile.imread(mask_path)
    props = regionprops(lm, spacing=spacing)
    props = [p for p in props if p.area >= MIN_VOXELS * spacing_um**3]

    centroids = np.array([p.centroid for p in props])
    organoid_center = centroids.mean(axis=0) if len(centroids) else np.zeros(3)
    z_extent = lm.shape[0] * spacing_um

    rows = []
    for i, p in enumerate(props):
        volume_um = p.area
        try:
            verts, faces, _, _ = marching_cubes(p.image.astype(np.uint8), level=0.5, spacing=spacing)
            surface_area_um = mesh_surface_area(verts, faces)
        except (RuntimeError, ValueError):
            surface_area_um = np.nan

        sphericity = (
            (np.pi ** (1 / 3)) * (6 * volume_um) ** (2 / 3) / surface_area_um
            if surface_area_um and surface_area_um > 0 else np.nan
        )

        ev = np.asarray(p.inertia_tensor_eigvals)  # descending, spacing-aware
        axis_lengths = []
        for ax in range(2, -1, -1):
            w = sum(v * -1 if j == ax else v for j, v in enumerate(ev))
            axis_lengths.append(np.sqrt(10 * max(w, 0)))
        major, medium, minor = axis_lengths  # full lengths, descending

        dist_to_center_um = float(np.linalg.norm(np.array(p.centroid) - organoid_center))
        neighbor_dists = np.linalg.norm(centroids - np.array(p.centroid), axis=1)
        neighborhood_density = int(((neighbor_dists > 0) & (neighbor_dists < 30)).sum())

        rows.append({
            "img_name": img_name,
            "spacing_x_um": spacing_um,
            "spacing_y_um": spacing_um,
            "spacing_z_um": spacing_um,
            "magnification": "25x" if "20241023" in img_name else "40x",
            "volume_um": volume_um,
            "surface_area_um": surface_area_um,
            "sphericity": sphericity,
            "solidity": p.solidity,
            "ellipsoid_axis_major_um": major,
            "ellipsoid_axis_medium_um": medium,
            "ellipsoid_axis_minor_um": minor,
            "aspect_ratio_minor_per_medium": minor / medium if medium else np.nan,
            "aspect_ratio_medium_per_major": medium / major if major else np.nan,
            "aspect_ratio_minor_per_major": minor / major if major else np.nan,
            "prolate_ratio": (major - medium) / major if major else np.nan,
            "oblate_ratio": (medium - minor) / medium if medium else np.nan,
            "relative_z_position": p.centroid[0] / z_extent if z_extent else np.nan,
            "distance_to_organoid_center_um": dist_to_center_um,
            "neighborhood_density": neighborhood_density,
            "nuc_count_per_organoid": len(props),
        })

    return pd.DataFrame(rows)


def extract_features_from_dir(mask_dir: str) -> pd.DataFrame:
    paths = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".tif")
    )
    return pd.concat([extract_features_from_mask(p) for p in paths], ignore_index=True)
