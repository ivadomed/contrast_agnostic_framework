#!/usr/bin/env bash
# Predict with the t2w baseline model over all 6 held-out test contrasts.
# Usage: bash 05_09_predict_t2w_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
