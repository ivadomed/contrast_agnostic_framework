#!/usr/bin/env bash
# Predict with the T2f/FLAIR-trained baseline (standard nnU-Net aug) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t2f — cross-contrast is the headline result).
#
# Usage:
#   bash 05_50_predict_t2f_baseline.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_50_predict_t2f_baseline.sh brats2024-glioma_t2f_baseline_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2f"
METHOD="t2f_baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fBaseline"
DATASET_ID="053"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
