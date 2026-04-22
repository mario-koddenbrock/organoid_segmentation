#!/bin/bash
#SBATCH --job-name=cpsam_autoseg
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%j_predict_autoseg.log
#SBATCH --error=logs/%j_predict_autoseg.log

REPO_DIR="/home/cluster_home/koddenb/workspace/organoid_segmentation"
CONDA_ENV="organoid_segmentation"
CONFIG="${REPO_DIR}/configs/predict_organoids_autoseg_cluster_config.json"

cd "${REPO_DIR}" || exit 1
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Config: ${CONFIG}"
echo "Start:  $(date)"
echo "---"

python predict_akps_nuclei.py --config "${CONFIG}" --skip-training

echo "---"
echo "End: $(date)"
