#!/usr/bin/env bash
# Predict with the atlas-liver-hcc baseline (no synthesis) on the held-out T1w test set.
# Usage: bash 05_02_predict_baseline.sh <RUN_ID> [FOLD]
# Example: bash 05_02_predict_baseline.sh atlas-liver-hcc_t1w_baseline_<TS> all
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerAtlasLiverHCCBaseline"
DATASET_ID="80"
CATEGORY="nnUNet"
export PREDICT_FOLDS="0 1 2"        # 3-fold models (CLAUDE.md fold policy)
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
