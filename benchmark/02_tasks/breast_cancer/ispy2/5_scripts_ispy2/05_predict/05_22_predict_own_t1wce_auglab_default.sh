#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t1wce-trained auglab_default model.
# Usage: bash 05_22_predict_own_t1wce_auglab_default.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
