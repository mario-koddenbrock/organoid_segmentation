"""Cross-domain comparison: train on manual (ground-truth) segmentation features,
evaluate on automated (CellposeSAM) segmentation features, and vice versa.

Features for both domains are computed identically via `extract_features.py` (not
Joshua's original CSV pipeline) so the comparison isn't confounded by differences
in feature-computation code between the two sources — only by genuine differences
in segmentation quality/consistency between manual and automated masks.

No CV is used here (mirrors the paper's own Section 3.3 methodology: fit on one
segmentation source, evaluate on the other) — with only 22 manually-annotated
organoids total, held-out CV would leave too few samples per fold to be meaningful.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.models import logreg_baseline, l0l2_logreg_model

FEATURE_COLS = [
    "volume_um", "surface_area_um", "sphericity", "solidity",
    "ellipsoid_axis_major_um", "ellipsoid_axis_medium_um", "ellipsoid_axis_minor_um",
    "aspect_ratio_minor_per_medium", "aspect_ratio_medium_per_major", "aspect_ratio_minor_per_major",
    "prolate_ratio", "oblate_ratio", "relative_z_position", "distance_to_organoid_center_um",
]
# neighborhood_density and nuc_count_per_organoid are excluded (per Joshua): neither
# is intrinsic to a single nucleus, and nuc_count is itself confounded with tumor
# status -- see analysis/akps_progression.py's module docstring for details.

MODEL_FACTORIES = {"logreg": logreg_baseline, "l0l2_logreg": l0l2_logreg_model}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_tumor"] = df["img_name"].str.contains("P013T").astype(int)
    df["organoid"] = df["img_name"]
    return df


def _get_Xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return X, df["is_tumor"].to_numpy()


def _aggregate_organoid(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("organoid")[FEATURE_COLS].mean()
    agg["is_tumor"] = df.groupby("organoid")["is_tumor"].first()
    return agg.reset_index()


def eval_cross_domain(train_df, test_df, model_factory, mode):
    """mode: 'nucleus', 'organoid_pred_agg', or 'organoid_data_agg'."""
    if mode == "organoid_data_agg":
        tr, te = _aggregate_organoid(train_df), _aggregate_organoid(test_df)
        Xtr, ytr = tr[FEATURE_COLS], tr["is_tumor"].to_numpy()
        Xte, yte = te[FEATURE_COLS], te["is_tumor"].to_numpy()
    else:
        Xtr, ytr = _get_Xy(train_df)
        Xte, yte = _get_Xy(test_df)

    model = model_factory()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    if mode == "organoid_pred_agg":
        pred_df = pd.DataFrame({
            "organoid": test_df["organoid"].to_numpy(), "proba": proba, "y_true": yte,
        })
        per_organoid = pred_df.groupby("organoid").agg(
            proba=("proba", "mean"), y_true=("y_true", "first")
        )
        return roc_auc_score(per_organoid["y_true"], per_organoid["proba"])
    return roc_auc_score(yte, proba)


def run_all(manual_df, auto_dfs: dict, model_names=("logreg", "l0l2_logreg"),
            modes=("nucleus", "organoid_pred_agg", "organoid_data_agg")):
    """auto_dfs: {trial_name: df}. Returns a long-format results DataFrame."""
    rows = []
    for trial, auto_df in auto_dfs.items():
        for model_name in model_names:
            factory = MODEL_FACTORIES[model_name]
            for mode in modes:
                auc_m2a = eval_cross_domain(manual_df, auto_df, factory, mode)
                auc_a2m = eval_cross_domain(auto_df, manual_df, factory, mode)
                rows.append({"trial": trial, "model": model_name, "mode": mode,
                             "direction": "manual_to_auto", "auc": auc_m2a})
                rows.append({"trial": trial, "model": model_name, "mode": mode,
                             "direction": "auto_to_manual", "auc": auc_a2m})
    return pd.DataFrame(rows)
