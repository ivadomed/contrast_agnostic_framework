#!/usr/bin/env bash
# BIDS -> nnUNet raw for toothfairy2 (Dataset110_ToothFairy2CBCT). Cheap (hard-links
# + dataset.json), but still dispatched through run_job so nothing heavy ever runs on
# a login node.
#   bash 02_00_convert.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"

run_job --name "tf2_convert" --gpus 0 --cpus 8 --mem 24G --time 02:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/convert_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export BIDS_ROOT='${BIDS_ROOT}' nnUNet_raw='${nnUNet_raw}'
        export NNUNET_DATASET_ID='${NNUNET_DATASET_ID}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/02_00_convert.py'
    "
