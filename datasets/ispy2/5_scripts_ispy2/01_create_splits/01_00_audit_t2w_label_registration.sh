#!/usr/bin/env bash
# Audit the registration-derived T2w lesion masks against the shared DICOM frame.
# Read-only; CPU-only run_job (1120 masks resampled — well over the login-node budget).
#   bash 01_create_splits/01_00_audit_t2w_label_registration.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${DATASET_ROOT}/4_splits_ispy2/_logs"; mkdir -p "${LOG_DIR}"

run_job --name ispy2_t2w_label_audit --gpus 0 --cpus 16 --mem 64G --time 02:00:00 \
        --log "${LOG_DIR}/01_00_audit_t2w_label_registration.log" --wait -- \
    env ISPY2_FOV_WORKERS=16 "${PROJECT_ROOT}/.venv/bin/python" \
        "${HERE}/01_00_audit_t2w_label_registration.py"

echo "--- log tail ---"; tail -n 60 "${LOG_DIR}/01_00_audit_t2w_label_registration.log"
