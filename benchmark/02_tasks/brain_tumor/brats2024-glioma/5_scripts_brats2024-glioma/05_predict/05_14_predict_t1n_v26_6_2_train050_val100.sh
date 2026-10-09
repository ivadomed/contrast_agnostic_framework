#!/usr/bin/env bash
# Predict with T1n V26_6_2 (50% train synth / 100% val synth) on the held-out BraTS test set.
#
# Usage:
#   bash 05_14_predict_t1n_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_14_predict_t1n_v26_6_2_train050_val100.sh brats2024-glioma_t1n_v26_6_2_train050_val100_20260622_044535

set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
