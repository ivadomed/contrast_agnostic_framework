#!/usr/bin/env bash
# Predict with srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) trained on CHAOS T2spir
# across all test modalities.
# Usage: bash 05_33_predict_t2spir_srcsm.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
