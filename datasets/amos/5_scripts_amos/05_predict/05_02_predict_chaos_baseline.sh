#!/usr/bin/env bash
# Predict on AMOS CT+MRI with the CHAOS-TRAINED baseline model (MR-only control).
# AMOS has no model of its own — see 05_01_predict_chaos_common.sh.
# Usage: bash 05_02_predict_chaos_baseline.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
CATEGORY="nnUNet"
RUN_ID="${1:-chaos_baseline_20260614_153230}"
source "$(dirname "$0")/05_01_predict_chaos_common.sh" "$RUN_ID" "${@:2}"
