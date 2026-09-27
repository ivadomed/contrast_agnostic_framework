#!/usr/bin/env bash
# Predict with V26_6_2 (train050/val100) trained on CHAOS T2spir across all test modalities.
# Usage: bash 05_22_predict_t2spir_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
DATASET_ID="61"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
