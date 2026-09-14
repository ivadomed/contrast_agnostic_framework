#!/usr/bin/env bash
# Predict with the PET-trained baseline model, over ct/pet/psma_ct/psma_pet.
# Usage: bash 05_09_predict_pet_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_pet.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerAutoPETBaseline"
CATEGORY="nnUNet"
DATASET_ID="120"   # fixed — imagesTs_* are consolidated under Dataset120, see 05_01 header
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
