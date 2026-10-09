#!/usr/bin/env bash
# Predict with the T2w SynthSeg (no EM) AugLab model on the held-out BraTS test set, all folds, across all contrasts.
# Model trained by 04_17_train_t2w_synthseg_noEM.sh (AugLabDefault trainer, Synthseg config).
#
# Usage:
#   bash 05_12_predict_t2w_synthseg_noEM.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_12_predict_t2w_synthseg_noEM.sh brats2024-glioma_t2w_synthseg_noEM_20260620_125442 all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_synthseg_noEM"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
