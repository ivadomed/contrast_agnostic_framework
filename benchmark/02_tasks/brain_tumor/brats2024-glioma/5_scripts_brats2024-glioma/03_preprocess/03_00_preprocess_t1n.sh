#!/usr/bin/env bash
# Plan and preprocess Dataset051_BraTS2024GliomaT1n, then install the shared
# splits_final.json from 4_splits_brats2024-glioma/ into the preprocessed dir.
#
# NOTE: t1n is the primary/default modality (predates the numbered 03_preprocess/
# convention adopted for t2w/t2f/t1c) — its preprocessing was originally run ad hoc
# and this script did not exist. Added for parity/reproducibility; the live
# Dataset051 preprocessed dir already exists, re-running is idempotent
# (nnUNetv2_plan_and_preprocess) but not required.
#
# Usage:
#   bash 03_00_preprocess_t1n.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

DATASET_ID="051"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
if [ -z "${_DS_NAME}" ]; then
    echo "ERROR: Dataset${DATASET_ID}_* not found in ${nnUNet_raw}" >&2
    echo "       Run 02_01_convert_t1n.py first." >&2
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Preprocessing ${_DS_NAME} …"

run_job --name "preprocess_brats_t1n" --gpus 1 --slot 0 --wait \
    --log "/tmp/preprocess_brats_t1n.log" -- \
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
