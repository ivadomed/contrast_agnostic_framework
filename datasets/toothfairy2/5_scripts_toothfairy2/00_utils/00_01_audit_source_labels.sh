#!/usr/bin/env bash
# Per-source-id label histogram over the whole release (label-only pass; see the .py
# for why the merged-class presence check is not sufficient).
#   bash 00_01_audit_source_labels.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
cd "${PROJECT_ROOT}"
TF2_STAGE="${TF2_STAGE:-${SCRATCH:-/scratch/p/paulh}/toothfairy2}"
export TF2_ZIP="${TF2_ZIP:-${TF2_STAGE}/0_raw/ToothFairy2.zip}"
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"

run_job --name "tf2_labaudit" --gpus 0 --cpus 16 --mem 64G --time 02:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/labaudit_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export TF2_ZIP='${TF2_ZIP}' BIDS_ROOT='${BIDS_ROOT}' TF2_WORKERS='${TF2_WORKERS:-16}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/00_01_audit_source_labels.py'
    "
