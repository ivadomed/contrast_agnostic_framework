#!/usr/bin/env bash
# Predict with the open-ms T1w baseline (no synthesis) across all test contrasts.
# Usage: bash 05_11_predict_t1w_baseline.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example: bash 05_11_predict_t1w_baseline.sh open-ms_t1w_baseline_<TS> all
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOpenMSBaseline"
DATASET_ID="71"
CATEGORY="nnUNet"
export PREDICT_FOLDS="0 1 2"        # 3-fold models (CLAUDE.md fold policy)
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
