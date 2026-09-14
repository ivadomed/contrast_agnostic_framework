#!/usr/bin/env bash
# Predict with the CT-trained baseline model, over ct/pet/psma_ct/psma_pet.
# Usage: bash 05_02_predict_ct_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerAutoPETBaseline"
CATEGORY="nnUNet"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
