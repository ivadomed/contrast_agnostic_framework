#!/usr/bin/env bash
# Predict with the CHAOS T2spir baseline across all test modalities.
# Usage: bash 05_21_predict_t2spir_baseline.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_21_predict_t2spir_baseline.sh chaos_t2spir_baseline_<TS> all
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
DATASET_ID="61"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
