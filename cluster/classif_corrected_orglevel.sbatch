#!/bin/bash
#SBATCH --job-name=classif_corrected_org
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_classif_corrected_org.log
#SBATCH --error=logs/%j_classif_corrected_org.log

REPO_DIR="/home/cluster_home/koddenb/workspace/organoid_segmentation"
CONDA_ENV="organoid_segmentation"

cd "${REPO_DIR}" || exit 1
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "---"

# Corrected feature units, full dataset (25x batch included -- see classif_corrected_perNucleus.sbatch)
python run_classification_comparison.py \
    --models tabpfn tabfm autogluon \
    --aggregate-organoid \
    --output-suffix _corrected

echo "---"
echo "End: $(date)"
