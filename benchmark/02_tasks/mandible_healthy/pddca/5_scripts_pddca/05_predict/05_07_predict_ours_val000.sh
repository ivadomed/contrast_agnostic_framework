#!/usr/bin/env bash
# Predict on the FOV-matched PDDCA CT test set with the toothfairy2 CBCT-trained
# OURS (val000) model (cross-MODALITY: CBCT -> CT). See 05_01_predict_common.sh.
# Usage: bash 05_07_predict_ours_val000.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDualVal"
CATEGORY="auglab"
DATASET_ID="${TF2_DATASET_ID}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
