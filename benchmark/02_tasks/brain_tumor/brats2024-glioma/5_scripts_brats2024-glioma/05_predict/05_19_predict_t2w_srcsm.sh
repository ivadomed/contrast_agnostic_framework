#!/usr/bin/env bash
# Predict with the T2w SRCSM model on the held-out BraTS test set, all folds, across all contrasts.
#
# Usage:
#   bash 05_19_predict_t2w_srcsm.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_19_predict_t2w_srcsm.sh brats2024-glioma_t2w_srcsm_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="srcsm"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
