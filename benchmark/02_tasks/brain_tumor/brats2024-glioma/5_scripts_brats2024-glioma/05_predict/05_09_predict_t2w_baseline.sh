#!/usr/bin/env bash
# Predict with the T2w baseline on the held-out BraTS test set, all folds, across all contrasts.
#
# Usage:
#   bash 05_09_predict_t2w_baseline.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_09_predict_t2w_baseline.sh brats2024-glioma_t2w_baseline_20260620_125115 all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wBaseline"
DATASET_ID="052"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
