#!/usr/bin/env bash
# Dispatch the BraTS-SSA 2024 BIDSify step through run_job (gzip-writes 475 files --
# light, but routed through Slurm for consistency, per project convention).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

run_job --name brats_ssa_bidsify --gpus 0 --slot 0 --time 00:20:00 --mem 8G \
    --log "${DATASET_ROOT}/00_utils_bidsify.log" --wait -- \
    .venv/bin/python "${DATASET_ROOT}/5_scripts_brats-ssa2024/00_utils/00_00_ingest_and_bidsify.py"
