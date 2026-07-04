#!/usr/bin/env bash
# nnUNet plan & preprocess for Dataset070_OpenMS_FLAIR, then install our custom
# patient-level CV splits (nnUNet would otherwise generate its own at first training).
# Usage: bash 03_00_preprocess.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

DATASET_ID="${DATASET_ID:-70}"
SPLITS_SRC="${SPLITS_DIR}/splits_final.json"

# CPU-only job (preprocessing needs no GPU) → schedules on the CPU partition without
# waiting behind the (often long) L40S queue.
run_job --name openms_preprocess --gpus 0 --cpus 8 --mem 32G --slot 0 --wait -- bash -c "
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
