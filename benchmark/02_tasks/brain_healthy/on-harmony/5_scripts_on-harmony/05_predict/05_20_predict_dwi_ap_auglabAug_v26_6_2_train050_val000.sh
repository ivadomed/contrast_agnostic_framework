#!/usr/bin/env bash
# Predict with the dwi_ap auglabAug_v26_6_2_train050_val000 model over all 6 held-out test contrasts.
# Usage: bash 05_20_predict_dwi_ap_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerOnHarmonyAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
