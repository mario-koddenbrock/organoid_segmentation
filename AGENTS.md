# Agent handoff notes

Read `docs/NUCLEUS_FEATURE_PIPELINE.md` before changing or running any morphology
classification analysis.

## Scientific constraints

- Use only `analysis.features.PURE_NUCLEUS_FEATURES` as model inputs.
- One row and one prediction must represent one nucleus. Do not aggregate feature
  rows or predicted probabilities to organoid level.
- Use `img_name`/`organoid` only as the grouping key for cross-validation. Nuclei
  from one image must never occur in both train and test folds.
- Do not use organoid context such as nucleus count, neighborhood density,
  relative position, distance to organoid center, acquisition metadata, spacing,
  objective, file paths, or identifiers as predictors.
- Physical size features must come from the recomputation script and retain the
  exact per-axis TIFF spacing. Never infer isotropy from a filename.

## Canonical workflow

1. On the cluster, run `recompute_features_from_cluster_masks.py` directly (CPU;
   no GPU and normally no SLURM). The optional sbatch file exists only as a
   fallback.
2. Copy its five CSV/JSON outputs into
   `data/recomputed_pure_nucleus_features/` locally. Data and results are ignored
   by git by design.
3. Validate metadata and the two explicitly imputed NCO batches as described in
   `docs/NUCLEUS_FEATURE_PIPELINE.md`.
4. Run `run_classification_comparison.py` and
   `run_pure_nucleus_notebook_experiments.py` from the repository root.

The two notebooks are historical exploration, not executable sources of truth:
they contain old table paths, old feature names, and organoid aggregation. Port a
notebook cell to the canonical modules before reusing it; do not run the notebooks
unchanged for final results.
