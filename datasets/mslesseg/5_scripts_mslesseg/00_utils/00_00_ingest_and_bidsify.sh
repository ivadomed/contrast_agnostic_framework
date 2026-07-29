#!/usr/bin/env bash
# Dispatch the MSLesSeg BIDSify step through run_job (light — pure filesystem hard-link
# ops over ~460 files — but routed through Slurm for consistency, per project convention).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

run_job --name mslesseg_bidsify --gpus 0 --slot 0 --time 00:15:00 --mem 8G \
    --log "${DATASET_ROOT}/00_utils_bidsify.log" --wait -- \
    .venv/bin/python "${DATASET_ROOT}/5_scripts_mslesseg/00_utils/00_00_ingest_and_bidsify.py"
