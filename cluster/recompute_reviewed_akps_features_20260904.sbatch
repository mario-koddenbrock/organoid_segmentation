#!/bin/bash
#SBATCH --job-name=features_reviewed_akps
#SBATCH --partition=Debug_node
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed/logs/%j_features.log
#SBATCH --error=/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed/logs/%j_features.log

set -euo pipefail

RUN_DIR="/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed"
DATA_ROOT="/scratch/koddenbrock/organoid_segmentation"

cd "${RUN_DIR}"
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate organoid_segmentation

python recompute_features_from_cluster_masks.py \
    --root "${DATA_ROOT}" \
    --output-dir "${DATA_ROOT}/results/reviewed_20260904/features" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --reviewed-images-dir "${RUN_DIR}/extracted/after_september2026" \
    --reviewed-predictions-root "${DATA_ROOT}/results/reviewed_20260904/predictions"
