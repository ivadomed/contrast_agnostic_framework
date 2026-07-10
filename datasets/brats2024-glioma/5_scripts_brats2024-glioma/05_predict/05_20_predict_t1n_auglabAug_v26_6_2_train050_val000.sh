#!/usr/bin/env bash
# Predict with T1n AugLabAug + V26_6_2 (50% train synth / 0% val synth) on the held-out BraTS test set.
#
# Usage:
#   bash 05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_<TS>

set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
