#!/usr/bin/env bash
# Build imagesTs_{flair,t1w,t2w}/labelsTs_{flair,t1w,t2w} from the BIDS tree.
# Run AFTER 03_preprocess/03_00_reorient_to_lps.sh (BIDS must be LPS first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name mslesseg_build_inputs --gpus 0 --slot 0 --time 00:15:00 --mem 8G \
    --log "${DATASET_ROOT}/2_nnUNet_mslesseg/build_test_inputs.log" --wait -- \
    .venv/bin/python "$(dirname "${BASH_SOURCE[0]}")/05_00_build_test_inputs.py"
