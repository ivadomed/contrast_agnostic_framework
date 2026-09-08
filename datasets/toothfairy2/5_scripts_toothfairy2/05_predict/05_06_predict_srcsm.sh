#!/usr/bin/env bash
# Predict on toothfairy2's OWN held-out CBCT test cases with the srcsm model.
# See 05_01_predict_common.sh.
# Usage: bash 05_06_predict_srcsm.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_CBCT}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
