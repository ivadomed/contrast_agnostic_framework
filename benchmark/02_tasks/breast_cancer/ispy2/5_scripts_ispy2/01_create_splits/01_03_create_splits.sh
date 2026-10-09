#!/usr/bin/env bash
# Create the ispy2 patient-level, FOV-stratified train/test partition + 3-fold CV
# splits. Reads 4_splits_ispy2/fov_variants.json (the single source of truth for the
# case list) and writes partition.json / splits_final.json, copying the latter into
# both preprocessed Dataset dirs.
#
# Light enough for the login node (it reads the per-patient T1wce masks to compute
# tumour burden for stratification, then does pure bookkeeping) — well inside the
# ~10 CPU-min / 4 GB allowance, so no run_job wrapper is needed here, unlike
# 01_02 which rewrites ~1250 volumes.
#
#   bash 01_create_splits/01_03_create_splits.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

LOG_DIR="${DATASET_ROOT}/4_splits_ispy2/_logs"
mkdir -p "${LOG_DIR}"

"${PROJECT_ROOT}/.venv/bin/python" \
    "$(dirname "$(readlink -f "$0")")/01_03_create_splits.py" \
    2>&1 | tee "${LOG_DIR}/01_03_create_splits.log"

echo "--- outputs ---"
ls -la "${DATASET_ROOT}/4_splits_ispy2/partition.json" \
       "${DATASET_ROOT}/4_splits_ispy2/splits_final.json"
