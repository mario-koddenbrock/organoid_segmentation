"""Apply a healthy-vs-tumor (P021N vs P013T) classifier to the AKPS isogenic CRC
progression series (NCO -> A -> AK -> AKP -> AKPS) to test whether the learned
"cancer score" trends monotonically with mutational stage.

Trains on three different feature sources for comparison: manual (ground-truth)
segmentation, auto (CellposeSAM) segmentation, and the two pooled together.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.models import logreg_baseline, l0l2_logreg_model

FEATURE_COLS = [
    "volume_um", "surface_area_um", "sphericity", "solidity",
    "ellipsoid_axis_major_um", "ellipsoid_axis_medium_um", "ellipsoid_axis_minor_um",
    "aspect_ratio_minor_per_medium", "aspect_ratio_medium_per_major", "aspect_ratio_minor_per_major",
    "prolate_ratio", "oblate_ratio", "relative_z_position", "distance_to_organoid_center_um",
    "neighborhood_density", "nuc_count_per_organoid",
]

MODEL_FACTORIES = {"logreg": logreg_baseline, "l0l2_logreg": l0l2_logreg_model}

# Biological ordering of the progression series.
LINE_ORDER = ["NCO", "A", "AK", "AKP", "AKPS"]
LINE_LABELS = {
    "NCO": "NCO\n(wild-type)",
    "A": "A\nAPC-KO",
    "AK": "AK\n+KRAS-G12D",
    "AKP": "AKP\n+TP53-KO",
    "AKPS": "AKPS\n+SMAD4-KO",
}


def load_manual(path="data/manual_vs_auto/manual_features.csv"):
    df = pd.read_csv(path)
    df["is_tumor"] = df["img_name"].str.contains("P013T").astype(int)
    df["organoid"] = df["img_name"]
    return df


def load_auto(path="data/manual_vs_auto/auto_trial005_FULL_features.csv"):
    df = pd.read_csv(path)
    df["is_tumor"] = df["img_name"].str.contains("P013T").astype(int)
    df["organoid"] = df["img_name"]
    return df


def load_akps(path="data/manual_vs_auto/akps_features.csv"):
    df = pd.read_csv(path)
    df["organoid"] = df["img_name"]
    return df


def _aggregate_organoid(df: pd.DataFrame, extra_cols=()) -> pd.DataFrame:
    agg = df.groupby("organoid")[FEATURE_COLS].mean()
    for c in extra_cols:
        agg[c] = df.groupby("organoid")[c].first()
    return agg.reset_index()


def score_akps(train_df, akps_df, model_name, mode):
    """Train on train_df (P021N/P013T), score akps_df. mode in
    {'nucleus','organoid_pred_agg','organoid_data_agg'}. Returns a DataFrame with
    one row per organoid: organoid, line, tumor_score (mean predicted P(tumor))."""
    factory = MODEL_FACTORIES[model_name]

    if mode == "organoid_data_agg":
        train_agg = _aggregate_organoid(train_df, extra_cols=["is_tumor"])
        akps_agg = _aggregate_organoid(akps_df, extra_cols=["line"])
        Xtr, ytr = train_agg[FEATURE_COLS], train_agg["is_tumor"].to_numpy()
        model = factory()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(akps_agg[FEATURE_COLS])[:, 1]
        return pd.DataFrame({
            "organoid": akps_agg["organoid"], "line": akps_agg["line"], "tumor_score": proba,
        })

    Xtr = train_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    ytr = train_df["is_tumor"].to_numpy()
    model = factory()
    model.fit(Xtr, ytr)

    Xte = akps_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(Xte)[:, 1]

    if mode == "nucleus":
        return pd.DataFrame({
            "organoid": akps_df["organoid"], "line": akps_df["line"], "tumor_score": proba,
        })
    elif mode == "organoid_pred_agg":
        tmp = pd.DataFrame({
            "organoid": akps_df["organoid"].to_numpy(), "line": akps_df["line"].to_numpy(),
            "proba": proba,
        })
        per_organoid = tmp.groupby("organoid").agg(
            line=("line", "first"), tumor_score=("proba", "mean")
        )
        return per_organoid.reset_index()
    else:
        raise ValueError(f"unknown mode {mode!r}")


def trend_stats(scores_df: pd.DataFrame):
    """Spearman correlation between progression-stage index (0..4) and tumor_score,
    at organoid granularity."""
    stage_idx = scores_df["line"].map({l: i for i, l in enumerate(LINE_ORDER)})
    rho, pval = spearmanr(stage_idx, scores_df["tumor_score"])
    return rho, pval


def run_all(train_sets: dict, akps_df: pd.DataFrame,
            model_names=("logreg", "l0l2_logreg"),
            modes=("nucleus", "organoid_pred_agg", "organoid_data_agg")):
    """train_sets: {name: df}. Returns (scores_long_df, summary_df)."""
    all_scores = []
    summary_rows = []
    for train_name, train_df in train_sets.items():
        for model_name in model_names:
            for mode in modes:
                scores = score_akps(train_df, akps_df, model_name, mode)
                scores["train_set"] = train_name
                scores["model"] = model_name
                scores["mode"] = mode
                all_scores.append(scores)

                rho, pval = trend_stats(scores)
                line_means = scores.groupby("line")["tumor_score"].mean().reindex(LINE_ORDER)
                summary_rows.append({
                    "train_set": train_name, "model": model_name, "mode": mode,
                    "spearman_rho": rho, "spearman_p": pval,
                    **{f"mean_{l}": line_means[l] for l in LINE_ORDER},
                })
    return pd.concat(all_scores, ignore_index=True), pd.DataFrame(summary_rows)
