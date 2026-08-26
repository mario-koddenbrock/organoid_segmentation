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

# Columns that are identifiers / acquisition metadata, not morphology features.
# Excluded from the model input matrix. `magnification`/`spacing_*_um` are dropped
# because the one 25x batch (20241023) is P021N-only, i.e. it would leak class
# identity through acquisition settings rather than real morphology.
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
]


def load_and_filter(csv_path: str) -> pd.DataFrame:
    """Load one trial's nucleus morphology CSV and apply Joshi's filters.

    Derives a binary label `is_tumor` (0 = P021N healthy, 1 = P013T tumor) from
    `img_name`.
    """
    df = pd.read_csv(csv_path)

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
