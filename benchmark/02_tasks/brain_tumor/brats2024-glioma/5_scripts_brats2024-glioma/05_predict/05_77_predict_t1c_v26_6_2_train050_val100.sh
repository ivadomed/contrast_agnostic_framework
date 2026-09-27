#!/usr/bin/env bash
# Predict with the T1c-trained V26_6_2 alone (ladder rung 5, real fill) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t1c — cross-contrast is the headline result).
#
# Usage:
#   bash 05_77_predict_t1c_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_77_predict_t1c_v26_6_2_train050_val100.sh brats2024-glioma_t1c_v26_6_2_train050_val100_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t1c"
METHOD="t1c_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100"
DATASET_ID="054"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
