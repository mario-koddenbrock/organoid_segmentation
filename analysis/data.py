"""Loading and filtering of per-nucleus morphology feature CSVs.

Filters follow the criteria supplied by Joshi (co-author) so that results here are
directly comparable to his logistic-regression baseline.
"""

import numpy as np
import pandas as pd

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

# `ellipsoid_r_axis_major/medium/minor` and `area`/`area_convex` are stored as RAW
# VOXEL values (no `_um` suffix), unlike `volume_um`/`surface_area_um`/
# `distance_to_organoid_center_um` which are already correctly converted to physical
# units (verified: `area * spacing_x_um * spacing_y_um * spacing_z_um == volume_um`
# exactly). Comparing the raw axis columns directly across the 25x (20241023) and
# 40x batches — which have different spacing_x_um (0.26 vs 0.324) — is comparing
# different physical scales as if they were the same value. This produced a false
# "+17%"/"91-96% batch-predictable" signal; ground-truth manual nuclei diameters
# (see notebooks/classification_investigation.ipynb) show <1% real difference
# between magnifications once correctly converted. RAW_AXIS_COLUMNS are corrected
# to microns in `load_and_filter` (see `*_um` columns added there) and the raw
# versions are excluded from the model feature matrix.
RAW_AXIS_COLUMNS = ["ellipsoid_r_axis_major", "ellipsoid_r_axis_medium", "ellipsoid_r_axis_minor"]
RAW_VOXEL_COUNT_COLUMNS = ["area", "area_convex"]  # redundant with volume_um + solidity once corrected

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
] + RAW_AXIS_COLUMNS + RAW_VOXEL_COUNT_COLUMNS


def load_and_filter(csv_path: str, exclude_25x: bool = False) -> pd.DataFrame:
    """Load one trial's nucleus morphology CSV and apply Joshi's filters.

    Derives a binary label `is_tumor` (0 = P021N healthy, 1 = P013T tumor) from
    `img_name`.

    Also corrects `ellipsoid_r_axis_major/medium/minor` from raw voxels to microns
    (new `*_um` columns; used by `get_feature_matrix` in place of the raw ones —
    see RAW_AXIS_COLUMNS above for why).

    `exclude_25x`: NOT recommended by default — kept only for reproducing the
    earlier (mistaken) analysis. The 25x batch (20241023, P021N-only) was
    originally suspected as a non-biological confound because raw-voxel axis
    columns differed ~17% between magnifications, but that was a units bug in
    this loader, not a real effect: ground-truth manual nuclei diameters differ
    <1% between magnifications, and a batch classifier using only correctly-scaled
    features performs at chance (~50%), not the earlier reported 91-96%.
    """
    df = pd.read_csv(csv_path)

    if exclude_25x:
        df = df[df["magnification"] != "25x"]

    for col in RAW_AXIS_COLUMNS:
        df[f"{col}_um"] = df[col] * df["spacing_x_um"]

    df = df[~df["img_name"].isin(BAD_IMAGES)]

    counts_per_image = df.groupby("img_name")["img_name"].transform("count")
    df = df[counts_per_image >= MIN_NUCLEI_PER_IMAGE]

    df = df[df["volume_um"] >= MIN_VOLUME_UM3]

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
    feature_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and c != "is_tumor"
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            f"Non-numeric/NaN values found in feature columns: "
            f"{X.columns[X.isna().any()].tolist()}"
        )
    y = df["is_tumor"].to_numpy()
    groups = df["organoid"].to_numpy()
    return X, y, groups


def aggregate_to_organoid_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-nucleus rows to one row per organoid (mean of numeric
    features), fixing the pseudoreplication of treating ~30 correlated nuclei
    from the same organoid as independent samples. Also removes within-organoid
    noise, at the cost of shrinking the dataset to ~55 rows total.
    """
    feature_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and c != "is_tumor" and c != "organoid"
    ]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    agg = numeric.groupby(df["organoid"]).mean()
    agg["is_tumor"] = df.groupby("organoid")["is_tumor"].first()
    agg["n_nuclei"] = df.groupby("organoid").size()
    return agg.reset_index()


def get_organoid_feature_matrix(agg_df: pd.DataFrame):
    """Feature matrix for an organoid-level dataframe (one row = one organoid,
    so plain StratifiedKFold is valid — no grouping needed)."""
    feature_cols = [
        c for c in agg_df.columns if c not in ("organoid", "is_tumor", "n_nuclei")
    ]
    X = agg_df[feature_cols]
    y = agg_df["is_tumor"].to_numpy()
    return X, y
