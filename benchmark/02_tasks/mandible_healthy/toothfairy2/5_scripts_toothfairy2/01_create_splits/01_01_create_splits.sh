#!/usr/bin/env bash
# Create toothfairy2's train/test partition + 3-fold CV splits (see the .py header
# for the dentition-burden stratification rationale). Reads only the BIDS conversion
# audit — no image I/O — so it is genuinely light enough for a login node, but it is
# routed through run_job anyway for consistency with every other stage.
#   bash 01_01_create_splits.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"

run_job --name "tf2_splits" --gpus 0 --cpus 2 --mem 8G --time 00:30:00 \
    --slot 0 --wait --log "${LOG_DIR}/splits_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        # Exported explicitly, never inherited: the job runs in a fresh sbatch
        # environment, and on TamIA these are the scratch overrides. Letting them
        # fall back to the repo-relative defaults would write the splits somewhere
        # training never reads.
        export BIDS_ROOT='${BIDS_ROOT}' SPLITS_DIR='${SPLITS_DIR}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export NNUNET_DATASET_ID='${NNUNET_DATASET_ID}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_01_create_splits.py'
    "
