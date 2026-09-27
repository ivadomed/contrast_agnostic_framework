#!/usr/bin/env bash
# Predict with the dwi_ap baseline model over all 6 held-out test contrasts.
# Usage: bash 05_15_predict_dwi_ap_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
