#!/usr/bin/env bash
# Re-derive the T2w lesion masks through the shared DICOM frame of reference.
# CPU-only run_job; writes NEW label files, never modifies the existing ones.
#   bash 01_create_splits/01_01_rederive_t2w_labels.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${DATASET_ROOT}/4_splits_ispy2/_logs"; mkdir -p "${LOG_DIR}"

run_job --name ispy2_t2w_label_rederive --gpus 0 --cpus 16 --mem 64G --time 02:00:00 \
        --log "${LOG_DIR}/01_01_rederive_t2w_labels.log" --wait -- \
    env ISPY2_FOV_WORKERS=16 "${PROJECT_ROOT}/.venv/bin/python" \
        "${HERE}/01_01_rederive_t2w_labels.py"

echo "--- log tail ---"; tail -n 60 "${LOG_DIR}/01_01_rederive_t2w_labels.log"
