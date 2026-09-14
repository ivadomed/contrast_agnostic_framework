#!/usr/bin/env bash
# Predict with the PET-trained OURS val100 checkpoint mirror.
# Usage: bash 05_15_predict_pet_ours_val100.sh <RUN_ID_with__val100_> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_pet.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAutoPETAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
