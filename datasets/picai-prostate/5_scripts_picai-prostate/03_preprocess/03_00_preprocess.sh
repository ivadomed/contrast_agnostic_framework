#!/usr/bin/env bash
# nnUNet plan & preprocess for Dataset080_PICAI_T2W, then install our custom
# patient-level CV splits (nnUNet would otherwise generate its own at first training).
# Usage: bash 03_00_preprocess.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

DATASET_ID="${DATASET_ID:-80}"
SPLITS_SRC="${SPLITS_DIR}/splits_final.json"

# CPU-only job (preprocessing needs no GPU) → schedules on the CPU partition without
# waiting behind the GPU queue.
run_job --name picai_preprocess_t2w --gpus 0 --cpus 32 --mem 128G --time 04:00:00 --slot 0 --wait -- bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${nnUNet_results}'
    cd '${PROJECT_ROOT}'
    .venv/bin/nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -np 16 --verify_dataset_integrity
"

# (Re)install our custom splits — must exist before training so nnUNet honours them.
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
cp "${SPLITS_SRC}" "${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
echo "Installed custom splits → ${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
