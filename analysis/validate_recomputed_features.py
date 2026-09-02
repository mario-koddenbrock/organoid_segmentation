"""Validate and visualize recomputed physical nucleus features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.features import PURE_NUCLEUS_FEATURES


def validate(df: pd.DataFrame, source: str) -> None:
    missing = [column for column in PURE_NUCLEUS_FEATURES if column not in df]
    if missing:
        raise ValueError(f"{source}: missing features {missing}")
    if df[PURE_NUCLEUS_FEATURES].isna().any().any():
        raise ValueError(f"{source}: NaNs in canonical features")
    expected_volume = df["voxel_count"] * df["voxel_volume_um3"]
    if not np.allclose(df["volume_um3"], expected_volume):
        raise ValueError(f"{source}: physical volume invariant failed")
    ratios = {
        "aspect_ratio_minor_per_medium": df["ellipsoid_axis_minor_um"] / df["ellipsoid_axis_medium_um"],
        "aspect_ratio_medium_per_major": df["ellipsoid_axis_medium_um"] / df["ellipsoid_axis_major_um"],
        "aspect_ratio_minor_per_major": df["ellipsoid_axis_minor_um"] / df["ellipsoid_axis_major_um"],
    }
    for column, expected in ratios.items():
        if not np.allclose(df[column], expected):
            raise ValueError(f"{source}: {column} invariant failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/recomputed_pure_nucleus_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/classification"))
    args = parser.parse_args()

    frames = []
    for path in sorted(args.data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        validate(frame, path.name)
        frame["table"] = path.stem
        frame["spacing_zyx_um"] = frame.apply(
            lambda row: f"{row.spacing_z_um:.3f}/{row.spacing_y_um:.3f}/{row.spacing_x_um:.3f}", axis=1
        )
        frames.append(frame)
    if not frames:
        raise ValueError(f"No CSV files in {args.data_dir}")
    data = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = data.groupby(["table", "spacing_zyx_um", "spacing_source"], dropna=False).agg(
        n_nuclei=("label", "size"),
        n_images=("img_name", "nunique"),
        volume_median_um3=("volume_um3", "median"),
        major_axis_median_um=("ellipsoid_axis_major_um", "median"),
        minor_axis_median_um=("ellipsoid_axis_minor_um", "median"),
        sphericity_median=("sphericity", "median"),
    ).reset_index()
    summary.to_csv(args.output_dir / "scaled_feature_plausibility_summary.csv", index=False)

    features = ["volume_um3", "ellipsoid_axis_major_um", "ellipsoid_axis_minor_um", "sphericity"]
    labels = ["Volume (µm³)", "Major axis (µm)", "Minor axis (µm)", "Sphericity"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    groups = list(data.groupby("table", sort=True))
    for ax, feature, label in zip(axes.flat, features, labels):
        values = [group[feature].clip(upper=group[feature].quantile(0.99)) for _, group in groups]
        ax.boxplot(values, tick_labels=[name for name, _ in groups], showfliers=False)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Physically scaled single-nucleus feature plausibility")
    fig.tight_layout()
    fig.savefig(args.output_dir / "scaled_feature_plausibility.png", dpi=160)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
