"""Experiment B: train directly on the AKPS progression series itself (not a
transfer from P021N/P013T) to test which morphology features best order/separate
the 5 mutational stages (NCO -> A -> AK -> AKP -> AKPS).

Two complementary views, both interpretable/sparse (consistent with l0l2_logreg
used elsewhere):
  - Ordinal regression: predict the stage index (0..4) as a continuous target via
    L0L2-constrained sparse linear regression (fastsparsegams, SquaredError loss)
    -- directly answers "how well, and with which features, does a linear
    combination of morphology track progression severity."
  - Multi-class classification: plain multinomial logistic regression (reference
    only, not sparse) -- answers "can we tell the 5 stages apart at all," with a
    confusion matrix showing which adjacent stages are hardest to distinguish.
"""

import numpy as np
import pandas as pd
import fastsparsegams as fsg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

from analysis.akps_progression import LINE_ORDER, FEATURE_COLS

STAGE_IDX = {line: i for i, line in enumerate(LINE_ORDER)}


def prep_nucleus_level(akps_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the ordinal target without aggregating nucleus rows."""
    out = akps_df.copy()
    out["stage_idx"] = out["line"].map(STAGE_IDX)
    return out


class SparseOrdinalRegressor:
    """L0L2-constrained sparse linear regression predicting stage index."""

    def __init__(self, max_support_size=5):
        self.max_support_size = max_support_size
        self.scaler = StandardScaler()
        self.fit_result = None
        self.best_lambda = None
        self.best_gamma = None
        self.n_features_selected = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)

        self.fit_result = fsg.fit(
            X_scaled, y, loss="SquaredError", penalty="L0L2", algorithm="CDPSI",
            max_support_size=self.max_support_size, num_gamma=5,
        )
        best_per_k = {}
        for g_idx, gamma in enumerate(self.fit_result.gamma):
            supports = self.fit_result.support_size[g_idx]
            lambdas = self.fit_result.lambda_0[g_idx]
            for l_idx, k in enumerate(supports):
                if k == 0 or k > self.max_support_size:
                    continue
                pred = self.fit_result.predict(X_scaled, lambda_0=lambdas[l_idx], gamma=gamma).ravel()
                r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
                if k not in best_per_k or r2 > best_per_k[k][0]:
                    best_per_k[k] = (r2, lambdas[l_idx], gamma)
        best_k = max(best_per_k, key=lambda k: best_per_k[k][0])
        _, self.best_lambda, self.best_gamma = best_per_k[best_k]
        self.n_features_selected = best_k
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
        return self.fit_result.predict(X_scaled, lambda_0=self.best_lambda, gamma=self.best_gamma).ravel()

    def selected_features(self, feature_names):
        coeffs = np.asarray(
            self.fit_result.coeff(lambda_0=self.best_lambda, gamma=self.best_gamma, include_intercept=True).todense()
        ).ravel()
        return {f: c for f, c in zip(feature_names, coeffs[1:]) if abs(c) > 1e-10}


def cv_evaluate(nuclei_df: pd.DataFrame, n_folds=5, random_state=0):
    """Nucleus-level CV, grouped by organoid to prevent biological leakage."""
    X = nuclei_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_stage = nuclei_df["stage_idx"].to_numpy()
    y_line = nuclei_df["line"].to_numpy()
    groups = nuclei_df["organoid"].to_numpy()

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_reg_pred = np.full(len(nuclei_df), np.nan)
    oof_clf_pred = np.full(len(nuclei_df), "", dtype=object)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_stage, groups)):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_stage, y_test_stage = y_stage[train_idx], y_stage[test_idx]
        y_train_line, y_test_line = y_line[train_idx], y_line[test_idx]

        reg = SparseOrdinalRegressor(max_support_size=5)
        reg.fit(X_train, y_train_stage)
        pred_stage = reg.predict(X_test)
        oof_reg_pred[test_idx] = pred_stage
        rho, _ = spearmanr(y_test_stage, pred_stage)
        r_pearson, _ = pearsonr(y_test_stage, pred_stage)
        mae = np.mean(np.abs(y_test_stage - pred_stage))

        clf = LogisticRegression(max_iter=2000)
        scaler = StandardScaler()
        clf.fit(scaler.fit_transform(X_train), y_train_line)
        pred_line = clf.predict(scaler.transform(X_test))
        oof_clf_pred[test_idx] = pred_line
        acc = accuracy_score(y_test_line, pred_line)

        fold_rows.append({
            "fold": fold, "reg_spearman": rho, "reg_pearson": r_pearson, "reg_mae": mae,
            "clf_accuracy": acc, "n_selected_features": reg.n_features_selected,
        })

    nuclei_df = nuclei_df.copy()
    nuclei_df["oof_pred_stage"] = oof_reg_pred
    nuclei_df["oof_pred_line"] = oof_clf_pred
    return pd.DataFrame(fold_rows), nuclei_df


def fit_full_and_select(nuclei_df: pd.DataFrame):
    """Refit the sparse ordinal regressor on ALL AKPS organoids for the final
    feature-selection story (not for held-out evaluation -- see cv_evaluate)."""
    X = nuclei_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = nuclei_df["stage_idx"].to_numpy()
    reg = SparseOrdinalRegressor(max_support_size=5)
    reg.fit(X, y)
    return reg, reg.selected_features(FEATURE_COLS)
