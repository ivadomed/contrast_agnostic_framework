#!/usr/bin/env bash
# Predict with the SynthSeg (noEM) AugLab model on the held-out BraTS test set.
# Model trained by 04_07_train_synthseg_noEM.sh (AugLabDefault trainer, SynthSeg config).
# Predictions → 8_results_brats2024-glioma/01_predictions/auglab/<RUN_ID>/fold{k}/{contrast}/.
#
# Usage:
#   bash 05_07_predict_synthseg_noEM.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Examples:
#   bash 05_07_predict_synthseg_noEM.sh synthseg_noEM            # all folds (fold→slot)
#   bash 05_07_predict_synthseg_noEM.sh synthseg_noEM all
#   SLOT=3 bash 05_07_predict_synthseg_noEM.sh synthseg_noEM 0   # one fold on slot 3

set -euo pipefail
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_predict_common.sh" "$@"
