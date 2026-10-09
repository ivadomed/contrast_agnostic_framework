#!/usr/bin/env bash
# Build the MR-T1 OOD arm by resampling MR INTO THE CT FRAME (see the .py header).
# The ground truth is NOT touched — this arm is scored against the same labelsTs_ct
# the CT arm uses, so the two columns differ only by image modality.
#   bash 01_04_prepare_mr_in_ct_frame.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HS_STAGE="${HS_STAGE:-${SCRATCH:-/scratch/p/paulh}/hanseg}"
export HANSEG_ZIP="${HANSEG_ZIP:-${HS_STAGE}/0_raw/HaN-Seg.zip}"
[ -f "${HANSEG_ZIP}" ] || { echo "missing ${HANSEG_ZIP}" >&2; exit 1; }
[ -d "${nnUNet_raw}/imagesTs_ct" ] || { echo "CT arm must exist first (imagesTs_ct)" >&2; exit 1; }
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"

run_job --name "hanseg_mr_ctframe" --gpus 0 --cpus 16 --mem 96G --time 04:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/mr_ctframe_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export HANSEG_ZIP='${HANSEG_ZIP}' BIDS_ROOT='${BIDS_ROOT}' nnUNet_raw='${nnUNet_raw}'
        export HANSEG_WORKERS='${HANSEG_WORKERS:-8}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_04_prepare_mr_in_ct_frame.py'
    "
