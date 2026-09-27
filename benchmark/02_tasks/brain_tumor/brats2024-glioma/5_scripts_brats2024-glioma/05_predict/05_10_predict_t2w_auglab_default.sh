#!/usr/bin/env bash
# Predict with the T2w AugLab default on the held-out BraTS test set, all folds, across all contrasts.
#
# Usage:
#   bash 05_10_predict_t2w_auglab_default.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_10_predict_t2w_auglab_default.sh brats2024-glioma_t2w_auglab_default_20260620_125306 all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_auglab_default"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
