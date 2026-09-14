#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t2w-trained synthseg_noEM model.
# Usage: bash 05_25_predict_own_t2w_synthseg_noEM.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
