#!/usr/bin/env bash
# Predict with the t1w v26_6_2 model (PALETTE alone, 50% train synth / 100% val synth; ladder rung 5)
# over all 6 held-out test contrasts. nnUNet category (not the auglabAug variant).
# Usage: bash 05_21_predict_t1w_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOnHarmonyV26_6_2_train050_val100"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
