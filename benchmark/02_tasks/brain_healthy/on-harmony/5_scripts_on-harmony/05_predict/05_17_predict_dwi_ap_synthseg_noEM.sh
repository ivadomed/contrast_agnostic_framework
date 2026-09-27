#!/usr/bin/env bash
# Predict with the dwi_ap synthseg_noEM model over all 6 held-out test contrasts.
# Usage: bash 05_17_predict_dwi_ap_synthseg_noEM.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
