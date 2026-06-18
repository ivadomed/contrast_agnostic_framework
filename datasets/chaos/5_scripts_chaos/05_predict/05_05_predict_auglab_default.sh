#!/usr/bin/env bash
# Predict with auglab_default (AugLab default aug, no synthesis) on the CHAOS internal
# test set across modalities. Models live under 01_predictions/auglab/<RUN_ID>/.
# Usage: bash 05_05_predict_auglab_default.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_05_predict_auglab_default.sh chaos_auglab_default_20260611_120000 all
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
