#!/bin/bash
# Check all predictions are complete, then zip into one archive.
# Usage: bash cluster/zip_predictions.sh [--force]
#   --force  zip even if predictions are incomplete

SCRATCH="/scratch/koddenbrock"
OUT="${SCRATCH}/predictions_all.zip"
FORCE=0
[ "$1" = "--force" ] && FORCE=1

ERRORS=0

# Count tif files in a directory (0 if dir missing)
count_tifs() { ls "$1"/*.tif 2>/dev/null | wc -l; }

# Check one (source_dir -> pred_dir) pair and print result
check() {
    local source="$1"
    local pred="$2"
    local label="$3"
    local expected actual
    expected=$(count_tifs "${source}")
    actual=$(count_tifs "${pred}")
    if [ "${actual}" -ge "${expected}" ] && [ "${expected}" -gt 0 ]; then
        printf "    OK      %-60s (%d/%d)\n" "${label}" "${actual}" "${expected}"
    else
        printf "    MISSING %-60s (%d/%d)\n" "${label}" "${actual}" "${expected}"
        ERRORS=$((ERRORS + 1))
    fi
}

# For predict_dirs configs: derive output path from source path
# Output structure: <pred_base>/<model>/<parent_dir>/<basename>/
check_explicit() {
    local source="$1" pred_base="$2" model="$3"
    local subdir parent pred_dir
    subdir=$(basename "${source}")
    parent=$(basename "$(dirname "${source}")")
    pred_dir="${pred_base}/${model}/${parent}/${subdir}"
    check "${source}" "${pred_dir}" "${parent}/${subdir}"
}

# ---------------------------------------------------------------------------
echo "============================================================"
echo " Checking AKPS Progression Organoids"
echo "============================================================"
AKPS_ROOT="${SCRATCH}/AKPS_Progression_Organoids"
AKPS_PRED="${AKPS_ROOT}/predictions"
AKPS_MODELS=(
    pretrained_cpsam
    trial_08_lr1e-4_wd1e-4
    trial_10_lr1e-4_wd1e-3
    trial_14_lr5e-4_wd1e-4
    trial_16_lr5e-4_wd1e-3
)
for model in "${AKPS_MODELS[@]}"; do
    echo "  Model: ${model}"
    for group_dir in "${AKPS_ROOT}"/*/; do
        group=$(basename "${group_dir}")
        [ "${group}" = "predictions" ] && continue
        for date_dir in "${group_dir}"*/; do
            [ -d "${date_dir}" ] || continue
            date=$(basename "${date_dir}")
            check "${date_dir}" "${AKPS_PRED}/${model}/${group}/${date}" "${group}/${date}"
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Checking Organoids_for_autosegmentation"
echo "============================================================"
AUTOSEG_PRED="${SCRATCH}/Organoids_for_autosegmentation/predictions"
AUTOSEG_DIRS=(
    "${SCRATCH}/Organoids_for_autosegmentation/20231108_Organoids_P021N/cropped_isotropic_images"
    "${SCRATCH}/Organoids_for_autosegmentation/20240220_Organoids_P013T/cropped_isotropic_images"
    "${SCRATCH}/Organoids_for_autosegmentation/20240305_Organoids_P013T/cropped_isotropic_images"
    "${SCRATCH}/Organoids_for_autosegmentation/20241009_Organoids_P013T/cropped_isotropic_images"
    "${SCRATCH}/Organoids_for_autosegmentation/20241023_Organoids_P021N/cropped_isotropic_images"
)
AUTOSEG_MODELS=(
    trial_08_lr1e-4_wd1e-4
    trial_10_lr1e-4_wd1e-3
    trial_14_lr5e-4_wd1e-4
    trial_16_lr5e-4_wd1e-3
)
for model in "${AUTOSEG_MODELS[@]}"; do
    echo "  Model: ${model}"
    for src in "${AUTOSEG_DIRS[@]}"; do
        check_explicit "${src}" "${AUTOSEG_PRED}" "${model}"
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Checking Organoids (labeled + unlabeled)"
echo "============================================================"
ORGANOIDS_PRED="${SCRATCH}/Organoids/predictions"
ORGANOIDS_DIRS=(
    "${SCRATCH}/Organoids/20231108_P021N_40xSil_Hoechst_SiRActin/images_cropped_isotropic"
    "${SCRATCH}/Organoids/20240220_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic"
    "${SCRATCH}/Organoids/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic"
    "${SCRATCH}/Organoids/20241009_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic"
    "${SCRATCH}/Organoids/20241023_P021N_25xSil_Hoechst_SirActin/images_cropped_isotropic"
    "${SCRATCH}/Organoids/unlabeled/20231108_Organoids_P021N"
    "${SCRATCH}/Organoids/unlabeled/20240220_Organoids_P013/cropped_isotropic_images"
    "${SCRATCH}/Organoids/unlabeled/20240305_Organoids_P013T/cropped_isotropic_images"
    "${SCRATCH}/Organoids/unlabeled/20241009_Organoids_P013T/cropped_isotropic_images"
    "${SCRATCH}/Organoids/unlabeled/20241023_Organoids_P021N/cropped_isotropic_images"
)
ORGANOIDS_MODELS=(
    trial_08_lr1e-4_wd1e-4
    trial_10_lr1e-4_wd1e-3
    trial_14_lr5e-4_wd1e-4
    trial_16_lr5e-4_wd1e-3
)
for model in "${ORGANOIDS_MODELS[@]}"; do
    echo "  Model: ${model}"
    for src in "${ORGANOIDS_DIRS[@]}"; do
        check_explicit "${src}" "${ORGANOIDS_PRED}" "${model}"
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if [ "${ERRORS}" -eq 0 ]; then
    echo " All predictions complete. Zipping..."
elif [ "${FORCE}" -eq 1 ]; then
    echo " ${ERRORS} incomplete folder(s) — zipping anyway (--force)."
else
    echo " ${ERRORS} incomplete folder(s). Aborting. Use --force to zip anyway."
    exit 1
fi
echo "============================================================"

cd "${SCRATCH}" || exit 1
echo "Start: $(date)"
zip -r "${OUT}" \
    AKPS_Progression_Organoids/predictions \
    Organoids_for_autosegmentation/predictions \
    Organoids/predictions

echo "Done:  $(date)"
echo "Size:  $(du -sh "${OUT}" | cut -f1)"
