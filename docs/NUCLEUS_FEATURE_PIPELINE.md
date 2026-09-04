# Physically scaled nucleus-feature pipeline

## Purpose and decision record

The original feature tables mixed intrinsic nucleus morphology with properties of
the surrounding organoid. Some length-like values were also represented in raw
pixels/voxels although the images were acquired or resampled at different physical
spacings. Those tables must not be used for the current prediction experiments.

The current contract is:

- only intrinsic 3D morphology of a single nucleus is a predictor;
- every training row and every prediction is one nucleus;
- no feature or prediction aggregation is performed;
- an organoid/image ID is used only to keep all its nuclei in one CV fold.

The notebooks `notebooks/classification_investigation.ipynb` and
`notebooks/akps_progression.ipynb` are maintained executable analyses using the
same canonical modules as `run_classification_comparison.py` and
`run_pure_nucleus_notebook_experiments.py`. They expose acquisition-scaling,
25×-objective, and imputed-NCO sensitivity checks.

## Canonical features

`analysis/features.py` is the machine-readable allowlist:

| Feature | Unit | Meaning |
|---|---:|---|
| `volume_um3` | µm³ | physical label volume |
| `surface_area_um2` | µm² | closed marching-cubes surface |
| `sphericity` | dimensionless | sphere-normalized volume/surface ratio |
| `solidity` | dimensionless | regionprops solidity |
| `ellipsoid_axis_major_um` | µm | equivalent ellipsoid full major axis |
| `ellipsoid_axis_medium_um` | µm | equivalent ellipsoid full middle axis |
| `ellipsoid_axis_minor_um` | µm | equivalent ellipsoid full minor axis |
| three `aspect_ratio_*` columns | dimensionless | ratios of ellipsoid axes |
| `prolate_ratio` | dimensionless | `(major - medium) / major` |
| `oblate_ratio` | dimensionless | `(medium - minor) / medium` |

Spacing, objective, voxel counts, identifiers, paths, line/dataset labels, nucleus
counts, neighborhood measures, relative position, and distance to organoid center
are documentation or grouping columns—not predictors.

## Recomputing from cluster masks

The cluster data root used on 2026-09-02 was:

```text
/scratch/koddenbrock/organoid_segmentation
```

The script was uploaded to:

```text
/home/cluster_home/koddenb/workspace/organoid_segmentation/recompute_features_from_cluster_masks.py
```

Run it directly in the `organoid_segmentation` conda environment; it is a
multi-process CPU workload and does not need a GPU:

```bash
cd /home/cluster_home/koddenb/workspace/organoid_segmentation
conda activate organoid_segmentation
python recompute_features_from_cluster_masks.py \
  --root /scratch/koddenbrock/organoid_segmentation \
  --output-dir /scratch/koddenbrock/organoid_segmentation/results/recomputed_pure_nucleus_features \
  --workers 8
```

`cluster/recompute_pure_nucleus_features.sbatch` is an optional fallback if direct
execution is inappropriate. SLURM was avoided for the completed run because the
queue was blocked; pending job `116238` was canceled.

The extractor reads `(z, y, x)` spacing from each source TIFF and passes it to
`skimage.measure.regionprops`. It pads each label crop before marching cubes so
the surface is closed, and excludes connected components below 30 voxels. It
preserves anisotropy: several AKPS stacks have approximately
`(0.300, 0.162, 0.162) µm`, despite `rescaled` in their names.

## Calibration audit and unresolved assumption

Observed P021N/P013T spacings include approximately 0.324 µm, 0.321 µm, and
0.260 µm isotropic voxels. AKPS includes `(0.300, 0.162, 0.162) µm` and
`(0.324, 0.324, 0.324) µm` data.

The following early 40× NCO batches lack usable TIFF calibration metadata:

- `20250722_NCO_images_rescaled_cropped`
- `20250804_NCO_images_rescaled_cropped`

They were assigned the explicit fallback `(0.300, 0.300, 0.300) µm`, based on
the related early AKPS processing. This is an assumption, not confirmed metadata.
Every affected row is marked `spacing_is_imputed=True` and has an
`imputed:<batch>` `spacing_source`. Confirm this spacing from microscope/acquisition
records before treating final biological conclusions as definitive; if corrected,
change `SPACING_FALLBACKS` and recompute all tables.

## Outputs from the completed run

The output directory contains a CSV plus JSON manifest for each set:

| CSV | Nuclei | Images with nuclei |
|---|---:|---:|
| `trial_005_p021n_p013t.csv` | 2,181 | 92 |
| `trial_005_akps.csv` | 6,656 | 220 |
| `trial_028_p021n_p013t.csv` | 2,271 | 90 |
| `trial_028_akps.csv` | 7,309 | 220 |
| `manual_p021n_p013t.csv` | 347 | 22 |

Total: 18,764 nuclei. Trial 028 has two masks without a retained nucleus (both
already known problematic images): `20240220...C_002...` and
`20240305...B_008...`.

The tables were copied locally to
`data/recomputed_pure_nucleus_features/`. `/data`, `/results`, and `/models` are
gitignored intentionally, so generated data and experiment artifacts are not in
the git commit. Recreate or copy them from the cluster before running analyses.

Validation performed after extraction found no NaNs in the 12 canonical feature
columns. `volume_um3 == voxel_count * voxel_volume_um3` and all stored aspect
ratios matched their source axes within floating-point tolerance.

## Analysis environment and commands

Package versions verified locally for the model comparison:

- `tabpfn==8.4.0`
- `tabfm==1.0.1`
- `autogluon.tabular==1.6.1`

Always activate `organoid_segmentation`. The machine's global Anaconda Python
currently has a NumPy/scikit-learn binary ABI mismatch and is not a valid runtime
for these analyses.

Run the model comparison:

```bash
python run_classification_comparison.py
```

Run the maintained versions of both notebook experiment families:

```bash
python run_pure_nucleus_notebook_experiments.py
```

Results produced before this physical-spacing recomputation are scientifically
stale and must not be reported. New outputs go to `results/classification/`.

For a standalone integrity check and physical-feature plausibility plot, run:

```bash
python -m analysis.validate_recomputed_features
```

## Reviewed AKPS replacement (2026-09-04)

Joshua supplied `20260904_AKPS_rescaled_cropped_reviewed.zip` after re-rescaling
and manually reviewing/cropping every progression line, including channel-order
corrections so nuclei are always channel 0. The archive SHA-256 is:

```text
2254eb4a5ad09632152a575445f895f52cec52fcdd2840008435abde7794d270
```

It is stored and extracted without replacing the previous dataset under:

```text
/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed/
```

The archive contains 226 TIFFs: NCO 51, A 45, AK 46, AKP 40, and AKPS 44.
Every image reports axes `ZCYX`, two channels, micron units, and effectively
isotropic `(z, y, x) = (0.324, 0.324, 0.324) µm` spacing. No spacing is imputed.
The complete audit is produced by `audit_reviewed_akps.py`.

SLURM array `141627` creates versioned Trial 005 and Trial 028 masks under
`results/reviewed_20260904/predictions`. Dependent job `141628` then creates
reviewed feature tables under `results/reviewed_20260904/features`. The job files
are `cluster/predict_reviewed_akps_20260904.sbatch` and
`cluster/recompute_reviewed_akps_features_20260904.sbatch`. Do not mix these masks
or tables with the previous AKPS outputs.
