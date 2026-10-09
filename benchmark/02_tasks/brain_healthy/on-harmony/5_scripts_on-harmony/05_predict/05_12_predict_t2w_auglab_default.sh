#!/usr/bin/env bash
# Predict with the t2w auglab_default model over all 6 held-out test contrasts.
# Usage: bash 05_12_predict_t2w_auglab_default.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
