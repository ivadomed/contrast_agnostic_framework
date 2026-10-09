#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t1wce-trained OURS model
# (auglabAug_v26_6_2, train050, val000 mirror of the DualVal run). Completes the
# t1wce own-eval 6-method suite (was missing 2026-09-06).
# Usage: bash 05_41_predict_own_t1wce_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerISPY2AugLabDualVal"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
