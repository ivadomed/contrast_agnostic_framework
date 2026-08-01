#!/usr/bin/env bash
# BIDSify T2-weighted Kidney MRI Segmentation (0_raw already downloaded+extracted).
#   bash 00_01_bidsify.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name kidney_t2w_bidsify --gpus 0 --slot 0 --mem 8G --time 00:30:00 \
    --log "${PROJECT_ROOT}/datasets/kidney-t2w/0_raw_kidney-t2w/_bidsify.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py"
