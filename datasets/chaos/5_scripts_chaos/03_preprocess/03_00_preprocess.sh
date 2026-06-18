#!/usr/bin/env bash
# nnUNet plan & preprocess for the CHAOS MR T1-in dataset, then install our custom
# CV splits (nnUNet would otherwise generate its own at first training).
# Usage: bash 03_00_preprocess.sh
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/../00_utils/env.sh"

DATASET_ID="${DATASET_ID:-60}"
SPLITS_SRC="${SPLITS_DIR}/splits_final.json"

set_slot 0 bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${nnUNet_results}'
    cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
    .venv/bin/nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
"

# (Re)install our custom splits — must exist before training so nnUNet honours them.
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
cp "${SPLITS_SRC}" "${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
echo "Installed custom splits → ${nnUNet_preprocessed}/${_DS_NAME}/splits_final.json"
