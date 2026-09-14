#!/usr/bin/env bash
# Predict on ispy2's own test cases with the T1WCE causal-ablation ladder rung 5
# model ("v26_6_2 alone", real-intensity fill -- see
# 04_19_train_t1wce_v26_6_2_train050_val100.sh). nnUNet-category (no
# NNUNET_RESULTS_BASE override at train time), TRAINER=ValSynth -- NOT to be
# confused with 05_41 (the headline OURS DualVal val000 mirror, auglab-category).
# Was missing 2026-09-06.
# Usage: bash 05_42_predict_own_t1wce_v26_6_2_train050_val100_ladder.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
CATEGORY="nnUNet"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
