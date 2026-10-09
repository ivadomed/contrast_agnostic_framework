#!/usr/bin/env bash
# Predict on toothfairy2's OWN held-out CBCT test cases with the ladder rung5 v26_6_2 alone (real fill) model.
# See 05_01_predict_common.sh.
# Usage: bash 05_12_predict_v26_6_2_train050_val100.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerToothFairy2AugLabValSynth"
CATEGORY="nnUNet"
DATASET_ID="${DATASET_ID_CBCT}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
