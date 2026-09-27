#!/usr/bin/env bash
# Predict on ispy2's OWN held-out test cases (t1wce+t2w items) with the ispy2
# t1wce-trained baseline model. See 05_20_predict_common.sh.
# Usage: bash 05_21_predict_own_t1wce_baseline.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerISPY2Baseline"
CATEGORY="nnUNet"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
