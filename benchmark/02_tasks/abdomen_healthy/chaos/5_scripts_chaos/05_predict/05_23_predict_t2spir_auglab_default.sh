#!/usr/bin/env bash
# Predict with AugLab default trained on CHAOS T2spir across all test modalities.
# Usage: bash 05_23_predict_t2spir_auglab_default.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
