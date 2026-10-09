#!/usr/bin/env bash
# Predict with the t1w synthseg_EM model over all 6 held-out test contrasts.
# Usage: bash 05_04_predict_t1w_synthseg_EM.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
