#!/usr/bin/env bash
# Build imagesTs_{t1n,t1c,t2w,t2f}/labelsTs_{t1n,t1c,t2w,t2f} from the BIDS tree.
# Run AFTER 03_preprocess/03_00_reorient_to_lps.sh (BIDS must be LPS first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name brats_ssa_build_inputs --gpus 0 --slot 0 --time 00:15:00 --mem 8G \
    --log "${DATASET_ROOT}/2_nnUNet_brats-ssa2024/build_test_inputs.log" --wait -- \
    .venv/bin/python "$(dirname "${BASH_SOURCE[0]}")/05_00_build_test_inputs.py"
