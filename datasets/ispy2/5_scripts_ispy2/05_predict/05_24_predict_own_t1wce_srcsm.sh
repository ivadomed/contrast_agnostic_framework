#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t1wce-trained srcsm model.
# Usage: bash 05_24_predict_own_t1wce_srcsm.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
