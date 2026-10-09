#!/usr/bin/env bash
# Predict with the T2w AugLab augmentation + V26_6_2 synthesis @25% train / 100% val on the held-out BraTS test set.
# All folds, across all contrasts.
#
# Usage:
#   bash 05_13_predict_t2w_auglabAug_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_13_predict_t2w_auglabAug_v26_6_2_train025_val100.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train025_val100_20260620_125531 all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
