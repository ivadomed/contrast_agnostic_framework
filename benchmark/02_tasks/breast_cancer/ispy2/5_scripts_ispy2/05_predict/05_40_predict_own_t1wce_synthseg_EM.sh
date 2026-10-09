#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t1wce-trained synthseg_EM model.
# Completes the t1wce own-eval 6-method suite (was missing 2026-09-06).
# Usage: bash 05_40_predict_own_t1wce_synthseg_EM.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
