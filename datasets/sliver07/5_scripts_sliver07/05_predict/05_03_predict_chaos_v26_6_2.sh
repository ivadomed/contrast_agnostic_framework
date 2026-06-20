#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS-TRAINED v26_6_2 model (best Mode B generator).
# SLIVER07 has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_03_predict_chaos_v26_6_2.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2"
TRAINER="nnUNetTrainerCHAOSV26_6_2"
CATEGORY="nnUNet"
RUN_ID="${1:-chaos_t1in_v26_6_2_train090_val000_20260614_205937}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
