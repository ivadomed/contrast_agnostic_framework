#!/usr/bin/env bash
# nnU-Net plan-and-preprocess for both totalseg-pelvic per-modality datasets (CT=130,
# MRI=131). Run via run_job, not the login node.
#
# Usage: bash 03_00_preprocess.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

declare -A SPLITS_SRC_BY_ID=(
    [130]="${DATASET_ROOT}/4_splits_totalseg-pelvic/ct/splits_final.json"
    [131]="${DATASET_ROOT}/4_splits_totalseg-pelvic/mri/splits_final.json"
)

for DATASET_ID in 130 131; do
    run_job --name "totalseg_pelvic_preprocess_${DATASET_ID}" --gpus 0 --cpus 16 --mem 64G \
        --time 08:00:00 --log "${RESULTS_DIR}/_logs/preprocess_${DATASET_ID}.log" --wait -- \
        .venv/bin/nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 3d_fullres --verify_dataset_integrity

    # (Re)install our custom CV splits — must exist before training so nnUNet honours
    # them instead of generating its own random split at first training. CT and MRI are
    # UNPAIRED (separate patient cohorts), so each modality installs its OWN split file
    # into its own Dataset ID's preprocessed dir (unlike chaos, which shares one split
    # file across two nnUNet datasets from the same patients).
    _DS_NAME="$(ls "${nnUNet_preprocessed}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
    cp "${SPLITS_SRC_BY_ID[$DATASET_ID]}" "${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
    echo "Installed custom splits for Dataset${DATASET_ID} -> ${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
done
