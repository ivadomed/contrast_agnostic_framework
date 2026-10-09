#!/usr/bin/env bash
# Predict on ispy2's own test cases with the ispy2 t2w-trained auglab_default model.
# Completes the t2w own-eval 6-method suite (was missing 2026-09-06).
# Usage: bash 05_44_predict_own_t2w_auglab_default.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
