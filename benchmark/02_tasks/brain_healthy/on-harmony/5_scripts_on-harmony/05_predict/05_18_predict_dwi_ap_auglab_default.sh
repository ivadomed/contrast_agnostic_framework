#!/usr/bin/env bash
# Predict with the dwi_ap auglab_default model over all 6 held-out test contrasts.
# Usage: bash 05_18_predict_dwi_ap_auglab_default.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
