#!/usr/bin/env bash
# Predict with the CT-trained srcsm model, over ct/pet/psma_ct/psma_pet.
# Usage: bash 05_06_predict_ct_srcsm.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
CATEGORY="auglab"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
