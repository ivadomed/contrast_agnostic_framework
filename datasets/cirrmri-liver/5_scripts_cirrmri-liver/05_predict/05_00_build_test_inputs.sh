#!/usr/bin/env bash
# Materialise the nnUNet test inputs for CIRRMRI-LIVER (t1 + t2 items).
#   bash 05_00_build_test_inputs.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name cirrmri_liver_build_test_inputs --gpus 0 --slot 0 --mem 8G --time 00:30:00 \
    --log "${PROJECT_ROOT}/datasets/cirrmri-liver/0_raw_cirrmri-liver/_build_test_inputs.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/05_00_build_test_inputs.py"
