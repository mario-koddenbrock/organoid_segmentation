"""Load physically scaled, per-nucleus morphology features."""

import re
import pandas as pd

from analysis.features import PURE_NUCLEUS_FEATURES

# Images excluded outright by Joshi (visually inspected bad segmentations).
BAD_IMAGES = [
    "20240305_P013T_B008",
    "20240305_P013T_C001",
    "20240305_P013T_C002",
    "20240305_P013T_C003",
    "20240305_P013T_C004",
]

MIN_NUCLEI_PER_IMAGE = 10
MIN_VOLUME_UM3 = 165.77
SPHERICITY_SOLIDITY_THRESHOLD = 0.65

RAW_AXIS_COLUMNS = ["ellipsoid_r_axis_major", "ellipsoid_r_axis_medium", "ellipsoid_r_axis_minor"]
RAW_VOXEL_COUNT_COLUMNS = ["area", "area_convex"]

# Excluded per Joshua's own feature-selection: neither is intrinsic to a single
# nucleus (neighborhood_density misses mitotic cells; nuc_count_per_organoid is
# itself confounded with tumor status -- cancer organoids carry more nuclei). See
# analysis/akps_progression.py's module docstring for the fuller rationale.
# `count` is an exact duplicate of `nuc_count_per_organoid` (verified: corr==1.0,
# identical per-organoid values) -- same column under two names in Joshi's CSV.
NON_INTRINSIC_COLUMNS = ["neighborhood_density", "nuc_count_per_organoid", "count"]

# Columns that are identifiers / acquisition metadata, not morphology features.
# Excluded from the model input matrix.
NON_FEATURE_COLUMNS = [
    "label",
    "img_name",
    "organoid",
    "date",
    "centroid",
    "organoid_center",
    "magnification",
    "spacing_x_um",
    "spacing_y_um",
    "spacing_z_um",
] + RAW_AXIS_COLUMNS + RAW_VOXEL_COUNT_COLUMNS + NON_INTRINSIC_COLUMNS


def load_and_filter(csv_path: str, exclude_25x: bool = False) -> pd.DataFrame:
    """Load one trial, normalize legacy names, and apply morphology filters."""
    df = pd.read_csv(csv_path)

    if exclude_25x:
        objective = df["objective"] if "objective" in df else df["magnification"]
        df = df[objective != "25x"]

    # Compatibility for old tables. New analyses must use the recomputed tables,
    # whose unit-bearing names are already present.
    if "volume_um3" not in df and "volume_um" in df:
        df["volume_um3"] = df["volume_um"]
    if "surface_area_um2" not in df and "surface_area_um" in df:
        df["surface_area_um2"] = df["surface_area_um"]

    for col in RAW_AXIS_COLUMNS:
        # Normalise the main CSV's ``ellipsoid_r_axis_*`` naming to the
        # canonical names emitted by the shared mask feature extractor.
        if col in df:
            canonical = col.replace("ellipsoid_r_axis_", "ellipsoid_axis_") + "_um"
            df[canonical] = df[col] * df["spacing_x_um"]

    def canonical_image_id(name):
        match = re.match(r"(\d{8})_(P021N|P013T).*?_([ABC])_?(\d{3}[a-z]?)$", str(name))
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}{match.group(4)}" if match else name

    df = df[~df["img_name"].map(canonical_image_id).isin(BAD_IMAGES)]
    if "organoid" not in df:
        df["organoid"] = df["img_name"]

    counts_per_image = df.groupby("img_name")["img_name"].transform("count")
    df = df[counts_per_image >= MIN_NUCLEI_PER_IMAGE]

    df = df[df["volume_um3"] >= MIN_VOLUME_UM3]

    bad_shape = (df["sphericity"] < SPHERICITY_SOLIDITY_THRESHOLD) & (
        df["solidity"] < SPHERICITY_SOLIDITY_THRESHOLD
    )
    df = df[~bad_shape]

    is_p021n = df["img_name"].str.contains("P021N")
    is_p013t = df["img_name"].str.contains("P013T")
    if not (is_p021n | is_p013t).all():
        unmatched = df.loc[~(is_p021n | is_p013t), "img_name"].unique()
        raise ValueError(f"img_name values without a known patient label: {unmatched}")
    df = df.assign(is_tumor=is_p013t.astype(int))

    return df.reset_index(drop=True)


def get_feature_matrix(df: pd.DataFrame):
    """Split a filtered dataframe into (X, y, groups) for grouped CV.

    Nuclei from the same organoid are correlated (same tissue, same imaging session),
    so CV folds MUST be grouped by `organoid` to avoid train/test leakage.
    """
    # Deliberately whitelist intrinsic single-nucleus measurements.  Inferring
    # features as "all numeric columns" previously admitted organoid-context
    # measurements such as relative position and distance to organoid centre.
    feature_cols = PURE_NUCLEUS_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing pure nucleus feature columns: {missing}")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            f"Non-numeric/NaN values found in feature columns: "
            f"{X.columns[X.isna().any()].tolist()}"
        )
    y = df["is_tumor"].to_numpy()
    groups = df["organoid"].to_numpy()
    return X, y, groups
