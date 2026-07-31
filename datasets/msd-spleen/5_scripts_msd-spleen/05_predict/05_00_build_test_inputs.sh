#!/usr/bin/env bash
# Materialise the nnUNet test inputs for MSD-SPLEEN (CT native hardlink-equivalent
# gzip copy, no resampling — 41 volumes, small enough to run on the login node's
# job queue directly).
#   bash 05_00_build_test_inputs.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name msd_spleen_build_test_inputs --gpus 0 --slot 0 --mem 8G --time 00:20:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/05_00_build_test_inputs.py"
