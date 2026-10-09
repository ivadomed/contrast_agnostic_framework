#!/usr/bin/env bash
# Build the FOV-matched HaN-Seg CT test set (see the .py header for the FOV-matching
# rationale, the leakage statement, and why the MR arm is deliberately not built).
#   bash 01_01_prepare_ct.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HS_STAGE="${HS_STAGE:-${SCRATCH:-/scratch/p/paulh}/hanseg}"
export HANSEG_ZIP="${HANSEG_ZIP:-${HS_STAGE}/0_raw/HaN-Seg.zip}"
[ -f "${HANSEG_ZIP}" ] || { echo "missing ${HANSEG_ZIP}" >&2; exit 1; }
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}" "${BIDS_ROOT}"

run_job --name "hanseg_prepare" --gpus 0 --cpus 8 --mem 96G --time 03:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/prepare_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export HANSEG_ZIP='${HANSEG_ZIP}' BIDS_ROOT='${BIDS_ROOT}' nnUNet_raw='${nnUNet_raw}'
        export HANSEG_WORKERS='${HANSEG_WORKERS:-6}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_01_prepare_ct.py'
    "
