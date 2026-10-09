#!/usr/bin/env bash
# Predict with the T1c-trained AugLab-default augmentation on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t1c — cross-contrast is the headline result).
#
# Usage:
#   bash 05_71_predict_t1c_auglab_default.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_71_predict_t1c_auglab_default.sh brats2024-glioma_t1c_auglab_default_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t1c"
METHOD="t1c_auglab_default"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cAugLabDefault"
DATASET_ID="054"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
