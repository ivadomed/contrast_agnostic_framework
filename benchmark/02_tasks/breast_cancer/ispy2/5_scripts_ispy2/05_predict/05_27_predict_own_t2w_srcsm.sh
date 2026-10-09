#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t2w-trained srcsm model.
# Usage: bash 05_27_predict_own_t2w_srcsm.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
