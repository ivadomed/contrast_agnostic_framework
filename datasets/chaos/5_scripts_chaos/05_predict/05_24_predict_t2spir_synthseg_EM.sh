#!/usr/bin/env bash
# Predict with SynthSeg+EM trained on CHAOS T2spir across all test modalities.
# Usage: bash 05_24_predict_t2spir_synthseg_EM.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
