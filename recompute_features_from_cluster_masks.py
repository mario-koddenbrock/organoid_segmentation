"""Recompute intrinsic 3D nucleus features from cluster masks and TIFF metadata."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import marching_cubes, mesh_surface_area, regionprops


PURE_FEATURES = [
    "volume_um3", "surface_area_um2", "sphericity", "solidity",
    "ellipsoid_axis_major_um", "ellipsoid_axis_medium_um", "ellipsoid_axis_minor_um",
    "aspect_ratio_minor_per_medium", "aspect_ratio_medium_per_major",
    "aspect_ratio_minor_per_major", "prolate_ratio", "oblate_ratio",
]

# These two early 40x NCO batches were resampled without preserving any TIFF
# calibration (resolution tags are 1/1 and ImageJ unit/spacing are absent).
# Other early AKPS batches processed by the same pipeline are 0.300 um isotropic.
# Keep this explicit and auditable instead of silently inferring from objective.
SPACING_FALLBACKS = {
    "20250722_NCO_images_rescaled_cropped": (0.3, 0.3, 0.3),
    "20250804_NCO_images_rescaled_cropped": (0.3, 0.3, 0.3),
}


def read_spacing_zyx(image_path: Path) -> tuple[tuple[float, float, float], str]:
    """Read physical spacing from ImageJ TIFF metadata and resolution tags."""
    with tifffile.TiffFile(image_path) as tif:
        meta = tif.imagej_metadata or {}
        if str(meta.get("unit", "")).lower() not in {"micron", "microns", "um", "µm"}:
            for batch, fallback in SPACING_FALLBACKS.items():
                if batch in image_path.parts:
                    return fallback, f"imputed:{batch}"
            raise ValueError(f"Missing micron unit in {image_path}")
        sz = float(meta["spacing"])
        page = tif.pages[0]

        def pixel_size(tag_name: str) -> float:
            value = page.tags[tag_name].value
            numerator, denominator = value
            return float(denominator) / float(numerator)

        sy = pixel_size("YResolution")
        sx = pixel_size("XResolution")
    spacing = np.asarray((sz, sy, sx), dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing {spacing} in {image_path}")
    # Preserve each axis exactly.  Several AKPS files called ``rescaled`` are
    # still anisotropic (for example z=0.300, y=x~0.162 um), so isotropy must
    # never be assumed from the filename.
    return tuple(spacing.tolist()), "tiff_metadata"


def image_name(path: Path) -> str:
    return re.sub(r"_cropped_isotropic(?:_nuclei-labels)?$|_rescaled_cropped$", "", path.stem)


def objective_from_name(path: Path) -> str | None:
    # Some cropped filenames omit the objective although their parent dataset
    # directory contains it, so inspect the complete path.
    match = re.search(r"(?:^|_)(25x|40x)", str(path), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def extract_one(pair: tuple[str, str, str, str]) -> list[dict]:
    mask_s, source_s, dataset, line = pair
    mask_path, source_path = Path(mask_s), Path(source_s)
    spacing, spacing_source = read_spacing_zyx(source_path)
    mask = tifffile.imread(mask_path)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}: {mask_path}")
    with tifffile.TiffFile(source_path) as tif:
        source_shape = tif.series[0].shape
        source_axes = tif.series[0].axes
    spatial_shape = tuple(source_shape[source_axes.index(a)] for a in "ZYX")
    if tuple(mask.shape) != spatial_shape:
        raise ValueError(f"Mask/source shape mismatch {mask.shape} vs {spatial_shape}: {mask_path}")

    rows = []
    for p in regionprops(mask, spacing=spacing):
        voxel_count = int(np.count_nonzero(p.image))
        if voxel_count < 30:
            continue
        volume = float(p.area)
        # Region crops touch their bounding box. Padding supplies the exterior
        # background needed for a closed marching-cubes surface.
        binary = np.pad(p.image.astype(np.uint8), 1)
        verts, faces, _, _ = marching_cubes(binary, level=0.5, spacing=spacing)
        surface = float(mesh_surface_area(verts, faces))
        sphericity = float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface)

        eig = np.asarray(p.inertia_tensor_eigvals, dtype=float)
        major = float(np.sqrt(10 * max(eig[0] + eig[1] - eig[2], 0)))
        medium = float(np.sqrt(10 * max(eig[0] - eig[1] + eig[2], 0)))
        minor = float(np.sqrt(10 * max(-eig[0] + eig[1] + eig[2], 0)))
        rows.append({
            "img_name": image_name(mask_path),
            "label": int(p.label),
            "dataset": dataset,
            "line": line or None,
            "mask_path": str(mask_path),
            "source_path": str(source_path),
            "objective": objective_from_name(source_path),
            "spacing_z_um": spacing[0],
            "spacing_y_um": spacing[1],
            "spacing_x_um": spacing[2],
            "spacing_source": spacing_source,
            "spacing_is_imputed": spacing_source.startswith("imputed:"),
            "voxel_volume_um3": float(np.prod(spacing)),
            "pixel_area_yx_um2": float(spacing[1] * spacing[2]),
            "spacing_anisotropy_ratio": float(max(spacing) / min(spacing)),
            "voxel_count": voxel_count,
            "volume_um3": volume,
            "surface_area_um2": surface,
            "sphericity": sphericity,
            "solidity": float(p.solidity),
            "ellipsoid_axis_major_um": major,
            "ellipsoid_axis_medium_um": medium,
            "ellipsoid_axis_minor_um": minor,
            "aspect_ratio_minor_per_medium": minor / medium,
            "aspect_ratio_medium_per_major": medium / major,
            "aspect_ratio_minor_per_major": minor / major,
            "prolate_ratio": (major - medium) / major,
            "oblate_ratio": (medium - minor) / medium,
        })
    return rows


def prediction_pairs(root: Path, trial: str, subset: str) -> list[tuple[str, str, str, str]]:
    pred = root / "results" / "predictions" / trial
    pairs = []
    if subset == "akps":
        datasets = ["NCO", "A", "AK", "AKP", "AKPS"]
        for line in datasets:
            for mask in sorted((pred / line).glob("**/*.tif")):
                source = root / "AKPS_Progression_Organoids" / line / mask.relative_to(pred / line)
                pairs.append((str(mask), str(source), f"AKPS/{line}", line))
    else:
        datasets = [p for p in pred.iterdir() if p.is_dir() and p.name.startswith("20")]
        for dataset_dir in sorted(datasets):
            is_auto_source = "_Organoids_" in dataset_dir.name
            source_base = root / ("Organoids_for_autosegmentation" if is_auto_source else "Organoids")
            for mask in sorted(dataset_dir.glob("**/*.tif")):
                source = source_base / dataset_dir.name / mask.relative_to(dataset_dir)
                pairs.append((str(mask), str(source), dataset_dir.name, ""))
    return pairs


def manual_pairs(root: Path) -> list[tuple[str, str, str, str]]:
    pairs = []
    for mask in sorted((root / "Organoids").glob("*/labelmaps/Nuclei/*.tif")):
        dataset = mask.parents[2].name
        source_name = mask.name.replace("_nuclei-labels.tif", ".tif")
        source = mask.parents[2] / "images_cropped_isotropic" / source_name
        pairs.append((str(mask), str(source), dataset, ""))
    return pairs


def reviewed_akps_pairs(
    images_dir: Path, predictions_dir: Path
) -> list[tuple[str, str, str, str]]:
    """Pair flat reviewed AKPS images with one model's predicted masks."""
    pairs = []
    for mask in sorted(predictions_dir.glob("*.tif")):
        source = images_dir / mask.name
        match = re.match(r"\d{8}_(NCO|AKPS|AKP|AK|A)_", mask.name)
        if not match:
            raise ValueError(f"Cannot infer AKPS line from {mask.name}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing reviewed source for {mask}: {source}")
        line = match.group(1)
        pairs.append((str(mask), str(source), f"reviewed_20260904/{line}", line))
    return pairs


def extract_set(pairs, output: Path, workers: int):
    if not pairs:
        raise ValueError(f"No masks selected for {output}")
    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(extract_one, pairs, chunksize=1):
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    summary = {
        "output": str(output), "n_masks": len(pairs), "n_nuclei": len(df),
        "features": PURE_FEATURES,
        "units": {
            "spacing_z_um": "um/voxel", "spacing_y_um": "um/voxel",
            "spacing_x_um": "um/voxel", "voxel_volume_um3": "um^3/voxel",
            "pixel_area_yx_um2": "um^2/pixel", "volume_um3": "um^3",
            "surface_area_um2": "um^2", "ellipsoid_axis_*_um": "um",
            "sphericity/solidity/aspect/prolate/oblate": "dimensionless",
        },
        "spacings": df[["spacing_z_um", "spacing_y_um", "spacing_x_um"]]
        .drop_duplicates().to_dict("records"),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--reviewed-images-dir", type=Path,
        help="Flat reviewed AKPS TIFF directory; switches to reviewed-data mode.",
    )
    parser.add_argument(
        "--reviewed-predictions-root", type=Path,
        help="Root containing trial_005 and trial_028 reviewed prediction trees.",
    )
    args = parser.parse_args()
    if args.reviewed_images_dir or args.reviewed_predictions_root:
        if not (args.reviewed_images_dir and args.reviewed_predictions_root):
            parser.error("Both reviewed-data arguments are required together")
        for trial in ("trial_005", "trial_028"):
            predictions = (
                args.reviewed_predictions_root / trial / "extracted" / "after_september2026"
            )
            extract_set(
                reviewed_akps_pairs(args.reviewed_images_dir, predictions),
                args.output_dir / f"{trial}_akps_reviewed_20260904.csv",
                args.workers,
            )
        return
    for trial in ("trial_005", "trial_028"):
        extract_set(prediction_pairs(args.root, trial, "p021n_p013t"),
                    args.output_dir / f"{trial}_p021n_p013t.csv", args.workers)
        extract_set(prediction_pairs(args.root, trial, "akps"),
                    args.output_dir / f"{trial}_akps.csv", args.workers)
    extract_set(manual_pairs(args.root), args.output_dir / "manual_p021n_p013t.csv", args.workers)


if __name__ == "__main__":
    main()
