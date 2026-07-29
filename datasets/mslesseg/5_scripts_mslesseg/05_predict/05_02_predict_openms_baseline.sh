#!/usr/bin/env bash
# Predict on MSLesSeg (FLAIR+T1w+T2w) with the open-ms BASELINE model (no synthesis).
# Usage: bash 05_02_predict_openms_baseline.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerOpenMSBaseline"
CATEGORY="nnUNet"
RUN_ID="${1:-open-ms_flair_baseline_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
