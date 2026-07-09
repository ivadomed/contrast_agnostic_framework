#!/usr/bin/env bash
# Predict with srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) on the CHAOS internal
# test set across modalities. Models live under 01_predictions/auglab/<RUN_ID>/.
# Usage: bash 05_32_predict_srcsm.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_32_predict_srcsm.sh chaos_t1in_srcsm_20260709_120000 all
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
