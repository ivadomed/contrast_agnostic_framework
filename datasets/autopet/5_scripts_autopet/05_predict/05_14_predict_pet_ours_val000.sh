#!/usr/bin/env bash
# Predict with the PET-trained OURS (auglabAug_v26_6_2_train050) val000 checkpoint
# mirror, over ct/pet/psma_ct/psma_pet.
# Usage: bash 05_14_predict_pet_ours_val000.sh <RUN_ID_with__val000_> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_pet.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAutoPETAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
