#!/usr/bin/env bash
# Predict on I-SPY2 (122 usable bilateral cases, t1wce+t2w items) with the
# ambl t1wce-trained auglabAug_v26_6_2_train050_val000 model. I-SPY2 has no model of its own — see
# 05_01_predict_common.sh.
# Usage: bash 05_07_predict_ambl_t1wce_auglabAug_v26_6_2_train050_val000.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAMBLAugLabDualVal"
CATEGORY="auglab"
export AMBL_TRAINING_CONTRAST="t1wce"
export AMBL_DATASET_ID="90"
RUN_ID="${1:-ambl_t1wce_auglabAug_v26_6_2_train050_val000_20260903_003721}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
