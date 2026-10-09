#!/usr/bin/env bash
# Predict with T2w V26_6_2 (50% train synth / 100% val synth) on the held-out BraTS test set.
# All folds, across all contrasts. (T2w sibling of 05_14_predict_t1n_v26_6_2_train050_val100.sh,
# nnUNet category — the plain v26_6_2 model, distinct from the auglabAug variant in 05_13.)
#
# Usage:
#   bash 05_17_predict_t2w_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_17_predict_t2w_v26_6_2_train050_val100.sh brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100"
DATASET_ID="052"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
