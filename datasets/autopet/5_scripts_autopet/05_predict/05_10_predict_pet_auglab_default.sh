#!/usr/bin/env bash
# Predict with the PET-trained auglab_default model, over ct/pet/psma_ct/psma_pet.
# Usage: bash 05_10_predict_pet_auglab_default.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_pet.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
CATEGORY="auglab"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
