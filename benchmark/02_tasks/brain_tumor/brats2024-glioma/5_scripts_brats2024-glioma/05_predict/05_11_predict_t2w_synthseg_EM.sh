#!/usr/bin/env bash
# Predict with the T2w SynthSeg+EM AugLab model on the held-out BraTS test set, all folds, across all contrasts.
# Model trained by 04_16_train_t2w_synthseg_EM.sh (AugLabDefault trainer, SynthSeg_EM config).
#
# Usage:
#   bash 05_11_predict_t2w_synthseg_EM.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_11_predict_t2w_synthseg_EM.sh brats2024-glioma_t2w_synthseg_EM_20260620_125354 all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_synthseg_EM"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
