#!/usr/bin/env bash
# Build the FOV-matched HaN-Seg MR-T1 test set, with the mandible label propagated
# from CT by mandible-local rigid registration.
#
# ⚠️ The GT this produces is REGISTRATION-PROPAGATED, not natively drawn — see the
# .py header. Every number from this arm carries registration error on top of model
# error and is NOT directly comparable to the CT arm.
#
#   bash 01_02_prepare_mr.sh
#   HANSEG_QC_TISSUE_FRAC=0.85 bash 01_02_prepare_mr.sh   # stricter QC
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HS_STAGE="${HS_STAGE:-${SCRATCH:-/scratch/p/paulh}/hanseg}"
export HANSEG_ZIP="${HANSEG_ZIP:-${HS_STAGE}/0_raw/HaN-Seg.zip}"
[ -f "${HANSEG_ZIP}" ] || { echo "missing ${HANSEG_ZIP}" >&2; exit 1; }
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}" "${BIDS_ROOT}"

# Registration is the expensive part (2 rigid multi-resolution passes x 42 cases),
# hence more CPU and a longer wall than the CT arm's pure crop+resample.
run_job --name "hanseg_prepare_mr" --gpus 0 --cpus 16 --mem 96G --time 06:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/prepare_mr_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export HANSEG_ZIP='${HANSEG_ZIP}' BIDS_ROOT='${BIDS_ROOT}' nnUNet_raw='${nnUNet_raw}'
        export HANSEG_WORKERS='${HANSEG_WORKERS:-8}'
        export HANSEG_QC_TISSUE_FRAC='${HANSEG_QC_TISSUE_FRAC:-0.80}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/01_02_prepare_mr.py'
    "
