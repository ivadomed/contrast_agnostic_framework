#!/usr/bin/env bash
# Predict on ispy2's own test cases with the T2W causal-ablation ladder rung 5
# model ("v26_6_2 alone", real-intensity fill). See
# 05_42_predict_own_t1wce_v26_6_2_train050_val100_ladder.sh (t1wce twin) for full
# rationale. nnUNet-category, TRAINER=ValSynth. Was missing 2026-09-06.
# Usage: bash 05_48_predict_own_t2w_v26_6_2_train050_val100_ladder.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
CATEGORY="nnUNet"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
