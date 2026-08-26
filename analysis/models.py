"""Thin wrappers around sklearn-compatible classifiers used in the model comparison.

Each factory returns a fresh, unfitted estimator exposing `.fit(X, y)` and
`.predict_proba(X)`.
"""

import os

from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()


def logreg_baseline():
    """L2-regularized logistic regression, standardized features.

    Reproducible baseline to sanity-check against the paper's reported ~0.8 AUC
    (constraint-based logistic regression on manually-selected features).
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, penalty="l2"),
    )


def l0l2_logreg_model(max_support_size=5, auc_threshold=0.8):
    """L0L2-constrained logistic regression via `fastsparsegams` (the current Python
    interface to L0Learn, by the same authors) — this is the actual method Joshi
    used ("L0L2-learn package" in the paper draft), not a plain sklearn baseline.

    Mirrors the paper's Fig 2A procedure: fit models with support size 1..
    `max_support_size`, and among those, use the smallest support size whose
    (training) AUC clears `auc_threshold`, falling back to the best-AUC support
    size if none clears it.
    """
    import numpy as np
    import fastsparsegams as fsg
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    class _L0L2LogReg:
        def __init__(self):
            self.scaler = StandardScaler()
            self.fit_result = None
            self.best_lambda = None
            self.best_gamma = None
            self.n_features_selected = None

        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=float))
            y_pm1 = np.where(np.asarray(y) == 1, 1, -1)

            self.fit_result = fsg.fit(
                X_scaled,
                y_pm1,
                loss="Logistic",
                penalty="L0L2",
                algorithm="CDPSI",
                max_support_size=max_support_size,
                num_gamma=5,
            )

            # Best (highest training AUC) solution found per support size k.
            best_per_k = {}
            for g_idx, gamma in enumerate(self.fit_result.gamma):
                supports = self.fit_result.support_size[g_idx]
                lambdas = self.fit_result.lambda_0[g_idx]
                for l_idx, k in enumerate(supports):
                    if k == 0 or k > max_support_size:
                        continue
                    proba = self.fit_result.predict(
                        X_scaled, lambda_0=lambdas[l_idx], gamma=gamma
                    ).ravel()
                    auc = roc_auc_score(y, proba)
                    if k not in best_per_k or auc > best_per_k[k][0]:
                        best_per_k[k] = (auc, lambdas[l_idx], gamma)

            if not best_per_k:
                raise RuntimeError("fastsparsegams found no non-trivial solutions")

            qualifying = [k for k, (auc, *_) in best_per_k.items() if auc >= auc_threshold]
            chosen_k = min(qualifying) if qualifying else max(
                best_per_k, key=lambda k: best_per_k[k][0]
            )
            _, self.best_lambda, self.best_gamma = best_per_k[chosen_k]
            self.n_features_selected = chosen_k
            return self

        def predict_proba(self, X):
            X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
            p1 = self.fit_result.predict(
                X_scaled, lambda_0=self.best_lambda, gamma=self.best_gamma
            ).ravel()
            return np.column_stack([1 - p1, p1])

    return _L0L2LogReg()


def tabpfn_model():
    from tabpfn import TabPFNClassifier

    os.environ.setdefault("TABPFN_NO_BROWSER", "true")
    if os.environ.get("TABPFN_TOKEN"):
        os.environ.setdefault("TABPFN_TOKEN", os.environ["TABPFN_TOKEN"])
    return TabPFNClassifier()


def tabfm_model():
    import torch
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

    # CPU inference is >100x slower than GPU for this model at ~1500-row context;
    # always prefer CUDA when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tabfm_v1_0_0_pytorch.load(model_type="classification", device=device)
    return TabFMClassifier(model=model)


def autogluon_model(label_col="is_tumor", time_limit=3600):
    """Returns a callable wrapper since AutoGluon's TabularPredictor needs a
    DataFrame with the label column attached, not a bare sklearn-style API.

    `time_limit` (seconds) bounds each fit's runtime — without it, best_quality
    (bagging + stacking over many model families) can run far longer than
    warranted on a ~1500-row training fold.
    """
    from autogluon.tabular import TabularPredictor

    class _AutoGluonWrapper:
        def __init__(self):
            self.label_col = label_col
            self.predictor = None

        def fit(self, X, y):
            import tempfile

            df = X.copy()
            df[self.label_col] = y
            self.predictor = TabularPredictor(
                label=self.label_col,
                eval_metric="roc_auc",
                path=tempfile.mkdtemp(prefix="autogluon_"),
                verbosity=1,
            ).fit(df, presets="best_quality", time_limit=time_limit)
            return self

        def predict_proba(self, X):
            proba = self.predictor.predict_proba(X)
            return proba.to_numpy()

    return _AutoGluonWrapper()


MODEL_FACTORIES = {
    "logreg": logreg_baseline,
    "l0l2_logreg": l0l2_logreg_model,
    "tabpfn": tabpfn_model,
    "tabfm": tabfm_model,
    "autogluon": autogluon_model,
}
