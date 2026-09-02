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

import re

import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.features import PURE_NUCLEUS_FEATURES
from analysis.models import logreg_baseline, l0l2_logreg_model

FEATURE_COLS = PURE_NUCLEUS_FEATURES
# neighborhood_density and nuc_count_per_organoid are excluded (per Joshua): neither
# is intrinsic to a single nucleus, and nuc_count is itself confounded with tumor
# status -- see analysis/akps_progression.py's module docstring for details.

MODEL_FACTORIES = {"logreg": logreg_baseline, "l0l2_logreg": l0l2_logreg_model}


def canonical_organoid_key(name: str) -> str:
    """Match short manual names to long automated-mask image names."""
    match = re.match(r"(\d{8})_(P021N|P013T).*?_([ABC])_?(\d{3}[a-z]?)$", str(name))
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}{match.group(4)}" if match else str(name)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_tumor"] = df["img_name"].str.contains("P013T").astype(int)
    df["organoid"] = df["img_name"]
    df["organoid_key"] = df["img_name"].map(canonical_organoid_key)
    return df


def _get_Xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return X, df["is_tumor"].to_numpy()


def eval_cross_domain(train_df, test_df, model_factory, mode):
    """Fit and evaluate individual nucleus rows; organoid aggregation is forbidden."""
    if mode != "nucleus":
        raise ValueError("Only nucleus-level prediction is supported")
    Xtr, ytr = _get_Xy(train_df)
    Xte, yte = _get_Xy(test_df)

    model = model_factory()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    return roc_auc_score(yte, proba)


def run_all(manual_df, auto_dfs: dict, model_names=("logreg", "l0l2_logreg"),
            modes=("nucleus",)):
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
