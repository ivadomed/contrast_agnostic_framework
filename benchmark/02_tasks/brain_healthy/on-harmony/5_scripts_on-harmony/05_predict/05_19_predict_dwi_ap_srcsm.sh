#!/usr/bin/env bash
# Predict with the dwi_ap srcsm model over all 6 held-out test contrasts.
# Usage: bash 05_19_predict_dwi_ap_srcsm.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
