#!/usr/bin/env bash
# Predict with v26_6_2_train050_val100 (t2w-trained, pansegdata) on the held-out test set: t1wce / t2w, all folds.
# Usage: bash 05_23_predict_t2w_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_25_run_all_predict_t2w.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerPANSEGDATAAugLabValSynth"
CATEGORY="nnUNet"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
