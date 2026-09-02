"""Apply a healthy-vs-tumor (P021N vs P013T) classifier to the AKPS isogenic CRC
progression series (NCO -> A -> AK -> AKP -> AKPS) to test whether the learned
"cancer score" trends monotonically with mutational stage.

Trains on three different feature sources for comparison: manual (ground-truth)
segmentation, auto (CellposeSAM) segmentation, and the two pooled together.

`neighborhood_density` and `nuc_count_per_organoid` are excluded from FEATURE_COLS
(per Joshua): neither is an intrinsic property of a single nucleus, mitotic cells
are often missed by segmentation (biasing density estimates), and nuc_count is
itself confounded with tumor status (cancer organoids carry more nuclei). For the
AKPS series specifically, Joshua additionally found several *other* features
correlate with nuc_count_per_organoid within that line (larger/more proliferative
organoids -> systematically different nucleus morphology), which would otherwise
leak organoid-size information into organoid-level feature means. `load_akps`
removes this by stratified subsampling: nuclei are subsampled down to an equal
count per organoid (see `stratified_subsample_by_organoid`), so every organoid
contributes the same number of samples to any per-nucleus average.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.features import PURE_NUCLEUS_FEATURES
from analysis.models import logreg_baseline, l0l2_logreg_model

FEATURE_COLS = PURE_NUCLEUS_FEATURES

MODEL_FACTORIES = {"logreg": logreg_baseline, "l0l2_logreg": l0l2_logreg_model}

MIN_NUCLEI_PER_ORGANOID = 10  # matches analysis/data.py's Joshi-derived threshold


def stratified_subsample_by_organoid(df: pd.DataFrame, group_col: str = "organoid",
                                      min_n: int = MIN_NUCLEI_PER_ORGANOID,
                                      target_n: int | None = None,
                                      random_state: int = 0) -> pd.DataFrame:
    """Equalize the number of nuclei contributed per organoid.

    Drops organoids with fewer than `min_n` nuclei, then randomly subsamples
    (without replacement) every remaining organoid down to `target_n` nuclei
    (default: the smallest organoid remaining after the `min_n` filter). This
    makes nuc_count_per_organoid constant across the dataset, so it structurally
    cannot correlate with -- and bias -- any other per-nucleus-averaged feature.
    """
    counts = df.groupby(group_col).size()
    keep_organoids = counts[counts >= min_n].index
    df = df[df[group_col].isin(keep_organoids)]
    if target_n is None:
        target_n = int(df.groupby(group_col).size().min())

    rng = np.random.RandomState(random_state)
    parts = []
    for _, sub in df.groupby(group_col):
        if len(sub) > target_n:
            idx = rng.choice(sub.index.to_numpy(), size=target_n, replace=False)
            parts.append(sub.loc[idx])
        else:
            parts.append(sub)
    return pd.concat(parts).sort_index().reset_index(drop=True)

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
    return stratified_subsample_by_organoid(df)


def score_akps(train_df, akps_df, model_name, mode):
    """Train on P021N/P013T and return one prediction per AKPS nucleus."""
    if mode != "nucleus":
        raise ValueError("Only nucleus-level prediction is supported")
    factory = MODEL_FACTORIES[model_name]

    Xtr = train_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    ytr = train_df["is_tumor"].to_numpy()
    model = factory()
    model.fit(Xtr, ytr)

    Xte = akps_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(Xte)[:, 1]

    return pd.DataFrame({
        "organoid": akps_df["organoid"], "line": akps_df["line"], "tumor_score": proba,
    })


def trend_stats(scores_df: pd.DataFrame):
    """Spearman correlation between stage index and per-nucleus tumor score."""
    stage_idx = scores_df["line"].map({l: i for i, l in enumerate(LINE_ORDER)})
    rho, pval = spearmanr(stage_idx, scores_df["tumor_score"])
    return rho, pval


def run_all(train_sets: dict, akps_df: pd.DataFrame,
            model_names=("logreg", "l0l2_logreg"),
            modes=("nucleus",)):
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
