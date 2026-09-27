#!/usr/bin/env bash
# Plan and preprocess Dataset053_BraTS2024GliomaT2f, then install the shared
# splits_final.json from 4_splits_brats2024-glioma/ into the preprocessed dir.
#
# Usage:
#   bash 03_02_preprocess_t2f.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2f.sh"

DATASET_ID="053"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
if [ -z "${_DS_NAME}" ]; then
    echo "ERROR: Dataset${DATASET_ID}_* not found in ${nnUNet_raw}" >&2
    echo "       Run 02_03_convert_t2f.py first." >&2
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Preprocessing ${_DS_NAME} …"

run_job --name "preprocess_brats_t2f" --gpus 1 --slot 0 --wait \
    --log "/tmp/preprocess_brats_t2f.log" -- \
    bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${nnUNet_results}'
    cd '${PROJECT_ROOT}'
    .venv/bin/nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity -c 3d_fullres
"

echo "[$(date '+%H:%M:%S')] Preprocessing done. Installing splits_final.json …"

PREPROCESSED_DS="${nnUNet_preprocessed}/${_DS_NAME}"
if [ ! -d "${PREPROCESSED_DS}" ]; then
    echo "ERROR: Preprocessed dir not found: ${PREPROCESSED_DS}" >&2
    exit 1
fi

cp "${SPLITS_DIR}/splits_final.json" "${PREPROCESSED_DS}/splits_final.json"
echo "  Installed: ${PREPROCESSED_DS}/splits_final.json"

echo "[$(date '+%H:%M:%S')] Done."
