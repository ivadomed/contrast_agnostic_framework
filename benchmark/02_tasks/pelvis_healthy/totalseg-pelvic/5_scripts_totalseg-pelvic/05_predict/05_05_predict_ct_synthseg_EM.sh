#!/usr/bin/env bash
# Predict with the CT-trained synthseg_EM model, over ct/mri.
# Usage: bash 05_05_predict_ct_synthseg_EM.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
CATEGORY="auglab"
DATASET_ID="130"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
