#!/usr/bin/env bash
# Predict with the T2f/FLAIR-trained SynthSeg+EM augmentation on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t2f — cross-contrast is the headline result).
#
# Usage:
#   bash 05_52_predict_t2f_synthseg_EM.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_52_predict_t2f_synthseg_EM.sh brats2024-glioma_t2f_synthseg_EM_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2f"
METHOD="t2f_synthseg_EM"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fAugLabDefault"
DATASET_ID="053"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
