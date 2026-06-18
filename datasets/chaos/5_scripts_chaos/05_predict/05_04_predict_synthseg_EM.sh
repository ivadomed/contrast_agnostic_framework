#!/usr/bin/env bash
# Predict with synthseg_EM (AugLab + SynthSeg-EM synthesis) on the CHAOS internal
# test set across modalities. Models live under 01_predictions/auglab/<RUN_ID>/.
# Usage: bash 05_04_predict_synthseg_EM.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_04_predict_synthseg_EM.sh chaos_synthseg_EM_train100_val000_20260611_120000 all
set -euo pipefail
METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
