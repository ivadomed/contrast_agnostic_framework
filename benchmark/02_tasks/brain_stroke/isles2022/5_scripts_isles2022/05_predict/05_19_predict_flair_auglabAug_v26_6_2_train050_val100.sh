#!/usr/bin/env bash
# Predict with auglabAug_v26_6_2_train050_val100 (flair-trained, isles2022) on the held-out test set: dwi / adc / flair, all folds.
# Usage: bash 05_19_predict_flair_auglabAug_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_25_run_all_predict_flair.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_flair.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISLES2022AugLabDualVal"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_FLAIR}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
