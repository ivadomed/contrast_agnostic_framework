#!/usr/bin/env bash
# Predict with the MRI-trained baseline model, over ct/mri.
# Usage: bash 05_13_predict_mri_baseline.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_mri.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerTotalsegPelvicBaseline"
CATEGORY="nnUNet"
DATASET_ID="130"   # fixed — imagesTs_* consolidated under Dataset130, see 05_01 header
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
