#!/usr/bin/env bash
# Predict on the FOV-matched PDDCA CT test set with the toothfairy2 CBCT-trained
# synthseg_EM model (cross-MODALITY: CBCT -> CT). See 05_01_predict_common.sh.
# Usage: bash 05_05_predict_synthseg_EM.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${TF2_DATASET_ID}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
