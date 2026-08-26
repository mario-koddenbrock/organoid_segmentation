"""Compare logistic regression, TabPFN, TabFM and AutoGluon on the tumor-vs-healthy
nucleus classification task, on both trial005 and trial028 segmentation datasets.

Grouped 5-fold CV (grouped by organoid) so nuclei from the same organoid never
appear in both train and test.

Usage:
    python run_classification_comparison.py
    python run_classification_comparison.py --models logreg tabpfn
    python run_classification_comparison.py --trials 005
"""

import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from analysis.data import load_and_filter, get_feature_matrix
from analysis.models import MODEL_FACTORIES

DATA_DIR = "data/Nuclei_morpho_data_trial005_trial028"
OUT_DIR = "results/classification"
N_FOLDS = 5
RANDOM_STATE = 0


def run_cv(model_name, X, y, groups, n_folds=N_FOLDS):
    gkf = GroupKFold(n_splits=n_folds)
    aucs, accs = [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), (
            "group leakage between train and test fold"
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MODEL_FACTORIES[model_name]()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        proba_pos = proba[:, 1] if proba.ndim == 2 else proba

        auc = roc_auc_score(y_test, proba_pos)
        acc = accuracy_score(y_test, (proba_pos >= 0.5).astype(int))
        aucs.append(auc)
        accs.append(acc)
        print(f"    fold {fold}: AUC={auc:.3f} ACC={acc:.3f} "
              f"(n_train={len(train_idx)}, n_test={len(test_idx)})")

    return np.array(aucs), np.array(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", default=["005", "028"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES.keys()))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for trial in args.trials:
        csv_path = os.path.join(DATA_DIR, f"morphology_features_nuclei_{trial}.csv")
        print(f"\n=== trial {trial} ({csv_path}) ===")
        df = load_and_filter(csv_path)
        X, y, groups = get_feature_matrix(df)
        print(f"n_rows={len(df)} n_organoids={len(set(groups))} "
              f"n_features={X.shape[1]} tumor_frac={y.mean():.3f}")

        for model_name in args.models:
            print(f"  -- model: {model_name}")
            t0 = time.time()
            try:
                aucs, accs = run_cv(model_name, X, y, groups)
                status = "ok"
                err = ""
            except Exception as exc:
                print(f"    FAILED: {exc}")
                traceback.print_exc()
                aucs, accs = np.array([np.nan]), np.array([np.nan])
                status = "failed"
                err = str(exc)
            elapsed = time.time() - t0

            rows.append({
                "trial": trial,
                "model": model_name,
                "auc_mean": aucs.mean(),
                "auc_std": aucs.std(),
                "acc_mean": accs.mean(),
                "acc_std": accs.std(),
                "n_folds": len(aucs),
                "status": status,
                "error": err,
                "elapsed_sec": elapsed,
            })

        trial_rows = [r for r in rows if r["trial"] == trial]
        pd.DataFrame(trial_rows).to_csv(
            os.path.join(OUT_DIR, f"comparison_{trial}.csv"), index=False
        )

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, "comparison_all.csv"), index=False)
    print("\n=== Summary ===")
    print(results.to_string(index=False))

    plot_results(results)


def plot_results(results: pd.DataFrame):
    import matplotlib.pyplot as plt

    ok = results[results["status"] == "ok"]
    if ok.empty:
        print("No successful runs to plot.")
        return

    trials = sorted(ok["trial"].unique())
    models = list(MODEL_FACTORIES.keys())
    x = np.arange(len(models))
    width = 0.8 / len(trials)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, trial in enumerate(trials):
        sub = ok[ok["trial"] == trial].set_index("model").reindex(models)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            sub["auc_mean"],
            width,
            yerr=sub["auc_std"],
            capsize=3,
            label=f"trial {trial}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUC-ROC (grouped 5-fold CV)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.8, linestyle="--", color="gray", linewidth=1, label="Joshi's reported AUC")
    ax.set_title("Tumor vs. healthy nucleus classification: model comparison")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "comparison_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
