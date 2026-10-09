#!/usr/bin/env bash
# nnUNet plan & preprocess for the CHAOS MR T2spir dataset (Dataset061), then install
# our custom CV splits (same splits_final.json as T1in — same patient partition).
#
# Run this AFTER 02_01_convert_t2spir.py has populated Dataset061_CHAOS_MR_T2spir.
#
# Usage: bash 03_01_preprocess_t2spir.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
cd "${PROJECT_ROOT}"

DATASET_ID="${DATASET_ID:-61}"
SPLITS_SRC="${SPLITS_DIR}/splits_final.json"

run_job --name chaos_preprocess_t2spir --gpus 1 --slot 0 --wait -- bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${nnUNet_results}'
    cd '${PROJECT_ROOT}'
    .venv/bin/nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
"

# (Re)install our custom splits — must exist before training so nnUNet honours them.
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
cp "${SPLITS_SRC}" "${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
echo "Installed custom splits → ${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
