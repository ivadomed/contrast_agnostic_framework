#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t2w-trained OURS model
# (auglabAug_v26_6_2, train050, val000 mirror of the DualVal run).
# Usage: bash 05_28_predict_own_t2w_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerISPY2AugLabDualVal"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
