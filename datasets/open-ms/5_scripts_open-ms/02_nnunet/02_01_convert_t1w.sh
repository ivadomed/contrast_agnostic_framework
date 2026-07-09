#!/usr/bin/env bash
# Build the T1w nnU-Net raw dataset (Dataset071_OpenMS_T1W) from BIDS.
# Thin wrapper around 02_01_convert_t1w.py — small CPU-only job (NIfTI file copies +
# mask binarisation), dispatched through run_job on the CPU partition.
#
# Run AFTER 00_utils/00_01_bidsify.py (BIDS tree must exist) and
# 01_create_splits/01_01_create_splits.py (partition.json must exist).
#
# Usage: bash 02_01_convert_t1w.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env_t1w.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${RESULTS_DIR}/_logs"
run_job --name openms_convert_t1w --gpus 0 --cpus 4 --mem 16G --time 00:20:00 \
    --log "${RESULTS_DIR}/_logs/openms_convert_t1w.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/02_01_convert_t1w.py"
