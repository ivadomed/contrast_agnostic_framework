#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name ms3seg_build_inputs --gpus 0 --slot 0 --time 00:15:00 --mem 8G \
    --log "${DATASET_ROOT}/2_nnUNet_ms3seg/build_test_inputs.log" --wait -- \
    .venv/bin/python "$(dirname "${BASH_SOURCE[0]}")/05_00_build_test_inputs.py"
