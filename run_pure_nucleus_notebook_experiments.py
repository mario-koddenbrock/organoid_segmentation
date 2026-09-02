"""Re-run the two notebook analyses with pure nucleus features and no aggregation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from analysis.akps_progression import (
    FEATURE_COLS,
    LINE_ORDER,
    load_akps,
    load_auto,
    load_manual,
    run_all as run_akps_transfer,
)
from analysis.akps_stage_classifier import (
    cv_evaluate,
    fit_full_and_select,
    prep_nucleus_level,
)
from analysis.manual_vs_auto import load as load_cross_domain
from analysis.manual_vs_auto import run_all as run_cross_domain
from analysis.models import l0l2_logreg_model


OUT = Path("results/classification")
DATA = Path("data/recomputed_pure_nucleus_features")
OUT.mkdir(parents=True, exist_ok=True)


def run_manual_vs_auto():
    manual = load_cross_domain(DATA / "manual_p021n_p013t.csv")
    auto = {
        "trial005": load_cross_domain(DATA / "trial_005_p021n_p013t.csv"),
        "trial028": load_cross_domain(DATA / "trial_028_p021n_p013t.csv"),
    }
    # Cross-domain evaluation compares the exact 22 manually annotated images,
    # never the full automated cohort against the smaller manual cohort.
    auto = {name: df[df["organoid_key"].isin(manual["organoid_key"].unique())].copy()
            for name, df in auto.items()}
    result = run_cross_domain(manual, auto, modes=("nucleus",))
    result.to_csv(OUT / "manual_vs_auto_pure_nucleus.csv", index=False)

    plot = result.copy()
    plot["series"] = plot["trial"] + " / " + plot["direction"]
    ax = plot.set_index("series")["auc"].plot.bar(figsize=(8, 4), ylim=(0.45, 1.0))
    ax.set_ylabel("Nucleus-level AUC")
    ax.set_title("Manual/automated cross-domain transfer: pure nucleus features")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "manual_vs_auto_pure_nucleus.png", dpi=150)
    plt.close(ax.figure)
    return result


def selected_transfer_features(train_sets):
    rows = []
    for name, df in train_sets.items():
        X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        y = df["is_tumor"].to_numpy()
        model = l0l2_logreg_model().fit(X, y)
        coeffs = np.asarray(
            model.fit_result.coeff(
                lambda_0=model.best_lambda,
                gamma=model.best_gamma,
                include_intercept=True,
            ).todense()
        ).ravel()[1:]
        rows.extend(
            {"train_set": name, "feature": feature, "coefficient": coefficient}
            for feature, coefficient in zip(FEATURE_COLS, coeffs)
            if abs(coefficient) > 1e-10
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "akps_transfer_selected_features_pure_nucleus.csv", index=False)
    return result


def run_akps():
    manual = load_manual(DATA / "manual_p021n_p013t.csv")
    auto = load_auto(DATA / "trial_005_p021n_p013t.csv")
    train_sets = {
        "manual": manual,
        "auto": auto,
        "manual+auto": pd.concat([manual, auto], ignore_index=True),
    }
    akps = load_akps(DATA / "trial_005_akps.csv")

    scores, summary = run_akps_transfer(train_sets, akps, modes=("nucleus",))
    scores.to_csv(OUT / "akps_scores_pure_nucleus.csv", index=False)
    summary.to_csv(OUT / "akps_summary_pure_nucleus.csv", index=False)
    selected_transfer_features(train_sets)

    primary = scores[scores["model"] == "l0l2_logreg"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (train_name, sub) in zip(axes, primary.groupby("train_set", sort=False)):
        values = [sub.loc[sub["line"] == line, "tumor_score"] for line in LINE_ORDER]
        ax.boxplot(values, tick_labels=LINE_ORDER, showfliers=False)
        rho, _ = spearmanr(sub["line"].map({x: i for i, x in enumerate(LINE_ORDER)}), sub["tumor_score"])
        ax.set_title(f"{train_name} (rho={rho:.2f})")
        ax.set_xlabel("AKPS line")
    axes[0].set_ylabel("Per-nucleus P(tumor)")
    fig.suptitle("AKPS transfer: pure nucleus features, no aggregation")
    fig.tight_layout()
    fig.savefig(OUT / "akps_trend_pure_nucleus.png", dpi=150)
    plt.close(fig)

    nuclei = prep_nucleus_level(akps)
    folds, predictions = cv_evaluate(nuclei)
    folds.to_csv(OUT / "akps_stage_cv_folds_pure_nucleus.csv", index=False)
    predictions.to_csv(OUT / "akps_stage_oof_predictions_pure_nucleus.csv", index=False)

    _, selected = fit_full_and_select(nuclei)
    selected_df = pd.DataFrame(
        [{"feature": feature, "coefficient": coefficient} for feature, coefficient in selected.items()]
    ).sort_values("coefficient")
    selected_df.to_csv(OUT / "akps_stage_selected_features_pure_nucleus.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stage_values = [
        predictions.loc[predictions["line"] == line, "oof_pred_stage"] for line in LINE_ORDER
    ]
    axes[0].boxplot(stage_values, tick_labels=LINE_ORDER, showfliers=False)
    axes[0].set_ylabel("Out-of-fold predicted stage")
    axes[0].set_xlabel("True stage")
    cm = confusion_matrix(predictions["line"], predictions["oof_pred_line"], labels=LINE_ORDER)
    ConfusionMatrixDisplay(cm, display_labels=LINE_ORDER).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Nucleus-level five-class predictions")
    fig.suptitle("AKPS direct models: pure nucleus features, grouped by organoid")
    fig.tight_layout()
    fig.savefig(OUT / "akps_stage_classifier_pure_nucleus.png", dpi=150)
    plt.close(fig)
    return summary, folds, selected_df


def main():
    print("Pure nucleus features:", ", ".join(FEATURE_COLS))
    print("\nManual vs auto")
    print(run_manual_vs_auto().to_string(index=False))
    summary, folds, selected = run_akps()
    print("\nAKPS transfer")
    print(summary.to_string(index=False))
    print("\nAKPS direct CV")
    print(folds.to_string(index=False))
    print("\nAKPS direct selected features")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
