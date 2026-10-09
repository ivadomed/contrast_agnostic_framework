#!/usr/bin/env bash
# Predict on the FOV-matched HaN-Seg CT test set with the toothfairy2 CBCT-trained
# baseline model (cross-MODALITY: CBCT -> CT). See 05_01_predict_common.sh.
# Usage: bash 05_02_predict_baseline.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerToothFairy2Baseline"
CATEGORY="nnUNet"
DATASET_ID="${TF2_DATASET_ID}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
