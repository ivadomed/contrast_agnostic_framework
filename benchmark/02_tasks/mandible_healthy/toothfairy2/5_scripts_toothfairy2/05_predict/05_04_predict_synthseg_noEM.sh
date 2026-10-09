#!/usr/bin/env bash
# Predict on toothfairy2's OWN held-out CBCT test cases with the synthseg_noEM model.
# See 05_01_predict_common.sh.
# Usage: bash 05_04_predict_synthseg_noEM.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_CBCT}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
