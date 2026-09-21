#!/usr/bin/env bash
# Predict with the T2f/FLAIR-trained V26_6_2 alone (ladder rung 5, real fill) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t2f — cross-contrast is the headline result).
#
# Usage:
#   bash 05_57_predict_t2f_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_57_predict_t2f_v26_6_2_train050_val100.sh brats2024-glioma_t2f_v26_6_2_train050_val100_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2f"
METHOD="t2f_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fV26_6_2_train050_val100"
DATASET_ID="053"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
