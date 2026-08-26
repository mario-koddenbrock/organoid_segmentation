"""Compare logistic regression, TabPFN, TabFM and AutoGluon on the tumor-vs-healthy
nucleus classification task, on both trial005 and trial028 segmentation datasets.

Grouped 5-fold CV (grouped by organoid) so nuclei from the same organoid never
appear in both train and test.

Every run scores ALL applicable evaluation levels from a single set of model fits
(training is identical regardless of how predictions are later aggregated, so we
don't waste GPU time re-training just to re-score):
  - without --aggregate-organoid: both 'nucleus' (raw per-nucleus AUC) and
    'organoid_pred_agg' (predicted probabilities meaned per organoid before scoring)
  - with --aggregate-organoid: 'organoid_data_agg' only (features meaned per
    organoid before training, so there's nothing left to aggregate post-hoc)
Each eval level writes its own comparison_*_{perNucleus,predAgg,orglevel}.csv/plot.

Usage:
    python run_classification_comparison.py
    python run_classification_comparison.py --models logreg tabpfn
    python run_classification_comparison.py --trials 005
    python run_classification_comparison.py --exclude-25x --aggregate-organoid

To parallelize across separate SLURM jobs (one fold+trial per job/GPU), run a
single fold and merge afterwards:
    python run_classification_comparison.py --trials 005 --models autogluon --fold 0
    python run_classification_comparison.py --merge  # aggregate results/classification/folds/*.csv
"""

import argparse
import glob
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from analysis.data import (
    load_and_filter,
    get_feature_matrix,
    aggregate_to_organoid_level,
    get_organoid_feature_matrix,
)
from analysis.models import MODEL_FACTORIES

DATA_DIR = "data/Nuclei_morpho_data_trial005_trial028"
OUT_DIR = "results/classification"
FOLD_OUT_DIR = os.path.join(OUT_DIR, "folds")
N_FOLDS = 5
RANDOM_STATE = 0

EVAL_LEVEL_SUFFIX = {
    "nucleus": "perNucleus",
    "organoid_pred_agg": "predAgg",
    "organoid_data_agg": "orglevel",
}


def run_cv(model_name, X, y, groups, n_folds=N_FOLDS, only_fold=None):
    """Run CV and score EVERY applicable evaluation level from a single set of
    model fits (one fit per fold, not one fit per eval level — training is
    identical regardless of how predictions are later aggregated, so scoring
    multiple ways from the same fit avoids wasted duplicate training runs).

    If `groups` is None, X/y are already organoid-level (one row per organoid,
    i.e. features were aggregated before training) so plain StratifiedKFold is
    used and the only eval level is 'organoid_data_agg'. Otherwise GroupKFold on
    `groups` (nucleus-level data, grouped by organoid to prevent leakage) and
    two eval levels are scored per fold: 'nucleus' (raw per-nucleus AUC) and
    'organoid_pred_agg' (predicted probabilities meaned per organoid before
    scoring — contrasted with aggregating the *features* before training, see
    `aggregate_to_organoid_level` in analysis/data.py).

    If `only_fold` is given, fit/evaluate only that one fold (used to
    parallelize folds across separate SLURM jobs).

    Returns {eval_level: (aucs, accs)}.
    """
    if groups is None:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(X, y)
        eval_levels = ["organoid_data_agg"]
    else:
        cv = GroupKFold(n_splits=n_folds)
        splits = cv.split(X, y, groups)
        eval_levels = ["nucleus", "organoid_pred_agg"]

    results = {lvl: ([], []) for lvl in eval_levels}
    for fold, (train_idx, test_idx) in enumerate(splits):
        if only_fold is not None and fold != only_fold:
            continue
        if groups is not None:
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), (
                "group leakage between train and test fold"
            )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MODEL_FACTORIES[model_name]()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        proba_pos = proba[:, 1] if proba.ndim == 2 else proba

        fold_metrics = {}
        if groups is None:
            fold_metrics["organoid_data_agg"] = (y_test, proba_pos)
        else:
            fold_metrics["nucleus"] = (y_test, proba_pos)
            test_groups = groups[test_idx]
            pred_df = pd.DataFrame({
                "organoid": test_groups, "proba": proba_pos, "y_true": y_test,
            })
            per_organoid = pred_df.groupby("organoid").agg(
                proba=("proba", "mean"), y_true=("y_true", "first")
            )
            fold_metrics["organoid_pred_agg"] = (
                per_organoid["y_true"].to_numpy(), per_organoid["proba"].to_numpy()
            )

        msg = f"    fold {fold} (n_train={len(train_idx)}, n_test={len(test_idx)}):"
        for lvl, (eval_y, eval_proba) in fold_metrics.items():
            auc = roc_auc_score(eval_y, eval_proba)
            acc = accuracy_score(eval_y, (eval_proba >= 0.5).astype(int))
            results[lvl][0].append(auc)
            results[lvl][1].append(acc)
            msg += f" [{lvl}] AUC={auc:.3f} ACC={acc:.3f}"
        print(msg)

    return {lvl: (np.array(a), np.array(c)) for lvl, (a, c) in results.items()}


def load_data(trial, exclude_25x=False, aggregate_organoid=False):
    """Load one trial's data, applying Joshi's filters plus the optional
    25x-magnification exclusion and organoid-level aggregation."""
    csv_path = os.path.join(DATA_DIR, f"morphology_features_nuclei_{trial}.csv")
    df = load_and_filter(csv_path, exclude_25x=exclude_25x)
    if aggregate_organoid:
        agg = aggregate_to_organoid_level(df)
        X, y = get_organoid_feature_matrix(agg)
        groups = None
        n_units = len(agg)
    else:
        X, y, groups = get_feature_matrix(df)
        n_units = len(set(groups))
    return X, y, groups, len(df), n_units


def run_single_fold(trial, model_name, fold, exclude_25x=False, aggregate_organoid=False):
    """Run one (trial, model, fold) combination and write its own result CSV
    (one row per eval level scored from that single fit — see `run_cv`).
    Meant to be invoked as one task of a SLURM job array."""
    os.makedirs(FOLD_OUT_DIR, exist_ok=True)
    X, y, groups, n_rows, n_units = load_data(trial, exclude_25x, aggregate_organoid)
    print(f"=== trial {trial} model {model_name} fold {fold} "
          f"(n_rows={n_rows} n_units={n_units}) ===")

    t0 = time.time()
    try:
        cv_results = run_cv(model_name, X, y, groups, only_fold=fold)
        rows = [
            {"trial": trial, "model": model_name, "fold": fold, "eval_level": lvl,
             "auc": aucs[0], "acc": accs[0], "status": "ok", "error": ""}
            for lvl, (aucs, accs) in cv_results.items()
        ]
    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        rows = [{"trial": trial, "model": model_name, "fold": fold, "eval_level": "unknown",
                  "auc": np.nan, "acc": np.nan, "status": "failed", "error": str(exc)}]
    elapsed = time.time() - t0
    for r in rows:
        r["elapsed_sec"] = elapsed

    out_path = os.path.join(FOLD_OUT_DIR, f"{trial}_{model_name}_fold{fold}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def merge_fold_results():
    """Aggregate per-fold CSVs from `run_single_fold` into the standard
    mean/std comparison table (same shape as the non-parallel run produces)."""
    paths = sorted(glob.glob(os.path.join(FOLD_OUT_DIR, "*.csv")))
    if not paths:
        print(f"No fold result files found in {FOLD_OUT_DIR}")
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    rows = []
    for (trial, model_name, eval_level), g in df.groupby(["trial", "model", "eval_level"]):
        ok = g[g["status"] == "ok"]
        rows.append({
            "trial": trial,
            "model": model_name,
            "eval_level": eval_level,
            "auc_mean": ok["auc"].mean(),
            "auc_std": ok["auc"].std(),
            "acc_mean": ok["acc"].mean(),
            "acc_std": ok["acc"].std(),
            "n_folds": len(ok),
            "status": "ok" if len(ok) == len(g) else "partial",
            "error": "; ".join(g.loc[g["status"] != "ok", "error"].dropna().unique()),
            "elapsed_sec": g["elapsed_sec"].sum(),
        })
    results = pd.DataFrame(rows).sort_values(["trial", "model", "eval_level"])
    os.makedirs(OUT_DIR, exist_ok=True)
    for lvl, suffix in EVAL_LEVEL_SUFFIX.items():
        lvl_results = results[results["eval_level"] == lvl]
        if lvl_results.empty:
            continue
        lvl_results.to_csv(os.path.join(OUT_DIR, f"comparison_merged_{suffix}.csv"), index=False)
        print(f"\n=== Merged ({lvl}) ===")
        print(lvl_results.to_string(index=False))
        plot_results(lvl_results, suffix=f"_merged_{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", default=["005", "028"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--fold", type=int, default=None,
                         help="Run only this single fold (requires exactly one "
                              "trial and one model); writes to results/classification/folds/")
    parser.add_argument("--merge", action="store_true",
                         help="Aggregate results/classification/folds/*.csv and exit")
    parser.add_argument("--exclude-25x", action="store_true",
                         help="Drop the 25x-magnification batch (20241023, P021N-only), "
                              "a confirmed non-biological confound with the class label")
    parser.add_argument("--aggregate-organoid", action="store_true",
                         help="Collapse to one row per organoid (mean features) instead "
                              "of per-nucleus rows, fixing pseudoreplication")
    parser.add_argument("--output-suffix", default="",
                         help="Prefix tag for output filenames, e.g. '_no25x'. The "
                              "eval-level name (perNucleus/predAgg/orglevel) is "
                              "appended automatically — see EVAL_LEVEL_SUFFIX.")
    args = parser.parse_args()

    if args.merge:
        merge_fold_results()
        return

    if args.fold is not None:
        assert len(args.trials) == 1 and len(args.models) == 1, (
            "--fold requires exactly one --trials value and one --models value"
        )
        run_single_fold(args.trials[0], args.models[0], args.fold,
                         args.exclude_25x, args.aggregate_organoid)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for trial in args.trials:
        print(f"\n=== trial {trial} (exclude_25x={args.exclude_25x}, "
              f"aggregate_organoid={args.aggregate_organoid}) ===")
        X, y, groups, n_rows, n_units = load_data(
            trial, args.exclude_25x, args.aggregate_organoid
        )
        print(f"n_rows={n_rows} n_units={n_units} "
              f"n_features={X.shape[1]} tumor_frac={y.mean():.3f}")

        for model_name in args.models:
            print(f"  -- model: {model_name}")
            t0 = time.time()
            try:
                cv_results = run_cv(model_name, X, y, groups)
                status_per_level = {lvl: "ok" for lvl in cv_results}
                err = ""
            except Exception as exc:
                print(f"    FAILED: {exc}")
                traceback.print_exc()
                cv_results = {lvl: (np.array([np.nan]), np.array([np.nan]))
                               for lvl in (["organoid_data_agg"] if args.aggregate_organoid
                                           else ["nucleus", "organoid_pred_agg"])}
                status_per_level = {lvl: "failed" for lvl in cv_results}
                err = str(exc)
            elapsed = time.time() - t0

            for lvl, (aucs, accs) in cv_results.items():
                rows.append({
                    "trial": trial,
                    "model": model_name,
                    "eval_level": lvl,
                    "auc_mean": aucs.mean(),
                    "auc_std": aucs.std(),
                    "acc_mean": accs.mean(),
                    "acc_std": accs.std(),
                    "n_folds": len(aucs),
                    "status": status_per_level[lvl],
                    "error": err,
                    "elapsed_sec": elapsed,
                })

    results = pd.DataFrame(rows)
    for lvl, suffix in EVAL_LEVEL_SUFFIX.items():
        lvl_results = results[results["eval_level"] == lvl]
        if lvl_results.empty:
            continue
        out_suffix = f"{args.output_suffix}_{suffix}"
        for trial in args.trials:
            trial_rows = lvl_results[lvl_results["trial"] == trial]
            trial_rows.to_csv(
                os.path.join(OUT_DIR, f"comparison_{trial}{out_suffix}.csv"), index=False
            )
        lvl_results.to_csv(os.path.join(OUT_DIR, f"comparison_all{out_suffix}.csv"), index=False)
        print(f"\n=== Summary ({lvl}) ===")
        print(lvl_results.to_string(index=False))
        plot_results(lvl_results, suffix=out_suffix)


def plot_results(results: pd.DataFrame, suffix=""):
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
    ax.axhline(0.8, linestyle="--", color="gray", linewidth=1, label="0.8 AUC reference")
    ax.set_title("Tumor vs. healthy nucleus classification: model comparison")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"comparison_plot{suffix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
