#!/usr/bin/env bash
# Predict with the T2w AugLab augmentation + V26_6_2 synthesis @50% train / 0% val on the held-out BraTS test set.
# All folds, across all contrasts.
#
# Usage:
#   bash 05_21_predict_t2w_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_21_predict_t2w_auglabAug_v26_6_2_train050_val000.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
