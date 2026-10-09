#!/usr/bin/env bash
# Test whether the MR arm's propagated GT is actually locked onto the mandible, by
# displacing it by known amounts and watching the QC metrics respond (or not).
# See the .py header for how to read the result.
#   bash 01_03_validate_mr_registration.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"
run_job --name "hanseg_val_mr" --gpus 0 --cpus 8 --mem 48G --time 01:30:00 \
    --slot 0 --wait --log "${LOG_DIR}/validate_mr_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export BIDS_ROOT='${BIDS_ROOT}' VAL_N_CASES='${VAL_N_CASES:-0}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_03_validate_mr_registration.py'
    "
