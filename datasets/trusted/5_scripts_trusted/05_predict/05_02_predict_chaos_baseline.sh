#!/usr/bin/env bash
# Predict on TRUSTED (CT + US) with the CHAOS-TRAINED baseline model (contrast: t1in).
# TRUSTED has no model of its own — see 05_01_predict_common.sh. Default items: ct us.
# Usage: bash 05_02_predict_chaos_baseline.sh [CHAOS_RUN_ID] [FOLD] [ITEM ...]   (ITEM ∈ {ct,us})
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
CATEGORY="nnUNet"
RUN_ID="${1:-chaos_t1in_baseline_20260614_153230}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
