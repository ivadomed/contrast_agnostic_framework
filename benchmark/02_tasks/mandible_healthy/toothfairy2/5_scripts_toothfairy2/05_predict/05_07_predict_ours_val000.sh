#!/usr/bin/env bash
# Predict on toothfairy2's OWN held-out CBCT test cases with the OURS (val000 mirror) model.
# See 05_01_predict_common.sh.
# Usage: bash 05_07_predict_ours_val000.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDualVal"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_CBCT}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
