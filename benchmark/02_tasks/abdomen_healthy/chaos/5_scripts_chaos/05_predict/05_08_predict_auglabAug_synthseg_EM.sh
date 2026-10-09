#!/usr/bin/env bash
# Predict with AugLab+SynthSeg_EM augmentation on the CHAOS internal test set.
# Usage: bash 05_08_predict_auglabAug_synthseg_EM.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
