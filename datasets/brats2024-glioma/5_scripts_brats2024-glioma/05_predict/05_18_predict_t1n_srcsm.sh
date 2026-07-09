#!/usr/bin/env bash
# Predict with the T1n SRCSM model on the held-out BraTS test set.
# Results live under 8_results_brats2024-glioma/01_predictions/auglab/<RUN_ID>/.
#
# Usage:
#   bash 05_18_predict_t1n_srcsm.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Examples:
#   bash 05_18_predict_t1n_srcsm.sh brats2024-glioma_t1n_srcsm_<TS>
#   bash 05_18_predict_t1n_srcsm.sh brats2024-glioma_t1n_srcsm_<TS> all

set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
