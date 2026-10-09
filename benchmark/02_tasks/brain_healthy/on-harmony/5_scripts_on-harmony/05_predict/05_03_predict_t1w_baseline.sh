#!/usr/bin/env bash
# Predict with the t1w baseline model over all 6 held-out test contrasts.
# Usage: bash 05_03_predict_t1w_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
