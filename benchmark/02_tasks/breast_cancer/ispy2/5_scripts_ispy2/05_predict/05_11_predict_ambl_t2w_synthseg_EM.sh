#!/usr/bin/env bash
# Predict on I-SPY2 (122 usable bilateral cases, t1wce+t2w items) with the
# ambl t2w-trained synthseg_EM model. I-SPY2 has no model of its own — see
# 05_01_predict_common.sh.
# Usage: bash 05_11_predict_ambl_t2w_synthseg_EM.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerAMBLAugLabDefault"
CATEGORY="auglab"
export AMBL_TRAINING_CONTRAST="t2w"
export AMBL_DATASET_ID="91"
RUN_ID="${1:-ambl_t2w_synthseg_EM_20260903_003721}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
