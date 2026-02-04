#!/bin/bash
#SBATCH --job-name=cpsam_finetune
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/%j_finetune.log
#SBATCH --error=cluster/logs/%j_finetune.log

# --- Paths ---
REPO_DIR="/home/cluster_home/koddenb/workspace/organoid_segmentation"
DATA_DIR="/scratch/koddenbrock/Organoids"
CONDA_ENV="organoid_segmentation"

# --- Training hyperparameters ---
N_EPOCHS=100
LEARNING_RATE=5e-5
BATCH_SIZE=8
TEST_FRACTION=0.2
MIN_MASKS_PER_SLICE=1
MIN_PIXELS=8
MIN_TRAIN_MASKS=5
SAVE_DIR="${REPO_DIR}"
MODEL_NAME="cpsam_nuclei_finetuned"
SEED=42

# --- 40x folders only ---
FOLDERS=(
    "20231108_P021N_40xSil_Hoechst_SiRActin"
    "20240220_P013T_40xSil_Hoechst_SiRActin"
    "20240305_P013T_40xSil_Hoechst_SiRActin"
    "20241009_P013T_40xSil_Hoechst_SiRActin"
)

# --- Setup ---
cd "${REPO_DIR}" || exit 1
mkdir -p cluster/logs
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "---"

# --- Run finetuning ---
python finetune_nuclei.py \
    --data_dir "${DATA_DIR}" \
    --folders "${FOLDERS[@]}" \
    --n_epochs ${N_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --test_fraction ${TEST_FRACTION} \
    --min_masks_per_slice ${MIN_MASKS_PER_SLICE} \
    --min_pixels ${MIN_PIXELS} \
    --min_train_masks ${MIN_TRAIN_MASKS} \
    --save_dir "${SAVE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --seed ${SEED}

echo "---"
echo "End: $(date)"
