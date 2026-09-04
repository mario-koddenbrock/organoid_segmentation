#!/bin/bash
#SBATCH --job-name=predict_reviewed_akps
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed/logs/%A_%a_predict.log
#SBATCH --error=/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed/logs/%A_%a_predict.log

set -euo pipefail

REPO_DIR="/home/cluster_home/koddenb/workspace/organoid_segmentation"
RUN_DIR="/scratch/koddenbrock/organoid_segmentation/incoming/20260904_AKPS_reviewed"
CONFIGS=(
  "${RUN_DIR}/reviewed_akps_20260904_trial005.json"
  "${RUN_DIR}/reviewed_akps_20260904_trial028.json"
)

cd "${REPO_DIR}"
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate organoid_segmentation

python -u evaluate_all.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" --device cuda
