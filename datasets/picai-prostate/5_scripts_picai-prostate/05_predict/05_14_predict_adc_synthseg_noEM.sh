#!/usr/bin/env bash
# Predict with synthseg_noEM (ADC-trained) on the picai-prostate
# held-out test set — T2W/ADC/HBV, folds 0 1 2.
# Usage: bash $(basename 05_predict/05_14_predict_adc_synthseg_noEM.sh) <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_adc.sh"
PREDICT_DATASET_ID_DEFAULT="81"
METHOD="adc_synthseg_noEM"
TRAINER="nnUNetTrainerPICAIProstateAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
