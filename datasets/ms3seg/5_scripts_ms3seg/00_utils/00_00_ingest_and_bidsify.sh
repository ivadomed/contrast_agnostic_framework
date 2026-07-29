#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"
run_job --name ms3seg_bidsify --gpus 0 --slot 0 --time 00:20:00 --mem 8G \
    --log "${DATASET_ROOT}/00_utils_bidsify.log" --wait -- \
    .venv/bin/python "${DATASET_ROOT}/5_scripts_ms3seg/00_utils/00_00_ingest_and_bidsify.py"
