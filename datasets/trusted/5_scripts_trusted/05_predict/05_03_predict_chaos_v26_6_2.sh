#!/usr/bin/env bash
# Predict on TRUSTED (CT + US) with the CHAOS-TRAINED v26_6_2 model (contrast: t1in).
# TRUSTED has no model of its own — see 05_01_predict_common.sh. Default items: ct us.
# Usage: bash 05_03_predict_chaos_v26_6_2.sh [CHAOS_RUN_ID] [FOLD] [ITEM ...]   (ITEM ∈ {ct,us})
set -euo pipefail
METHOD="v26_6_2"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
CATEGORY="nnUNet"
RUN_ID="${1:-chaos_t1in_v26_6_2_train050_val100_20260615_213615}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
