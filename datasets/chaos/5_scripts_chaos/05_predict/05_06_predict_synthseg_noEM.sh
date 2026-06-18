#!/usr/bin/env bash
# Predict with synthseg_noEM (AugLab + SynthSeg synthesis, no EM) on the CHAOS internal
# test set across modalities. Models live under 01_predictions/auglab/<RUN_ID>/.
# Usage: bash 05_06_predict_synthseg_noEM.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_06_predict_synthseg_noEM.sh chaos_synthseg_noEM_train100_val000_20260611_120000 all
set -euo pipefail
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
