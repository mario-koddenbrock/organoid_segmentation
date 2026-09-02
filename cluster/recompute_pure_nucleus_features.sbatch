#!/bin/bash
#SBATCH --job-name=recompute_nucleus_features
#SBATCH --partition=Debug_node
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_recompute_nucleus_features.log
#SBATCH --error=logs/%j_recompute_nucleus_features.log

REPO_DIR="/home/cluster_home/koddenb/workspace/organoid_segmentation"
DATA_ROOT="/scratch/koddenbrock/organoid_segmentation"
OUTPUT_DIR="${DATA_ROOT}/results/recomputed_pure_nucleus_features"

cd "${REPO_DIR}" || exit 1
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate organoid_segmentation

python recompute_features_from_cluster_masks.py \
    --root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${SLURM_CPUS_PER_TASK}"
