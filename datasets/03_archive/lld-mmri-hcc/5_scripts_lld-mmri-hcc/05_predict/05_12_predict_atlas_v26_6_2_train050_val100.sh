#!/usr/bin/env bash
# Predict on the atlas-liver-hcc causal-ablation ladder rung 5 (v26_6_2 alone,
# real-intensity fill -- the causal pair vs rung 4's noise fill). See
# 05_02_predict_atlas_baseline.sh header for the general cross-dataset predict note.
# Usage: bash 05_12_predict_atlas_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabValSynth"
CATEGORY="nnUNet"
RUN_ID="${1:-atlas-liver-hcc_t1w_v26_6_2_train050_val100_20260829_165445}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
