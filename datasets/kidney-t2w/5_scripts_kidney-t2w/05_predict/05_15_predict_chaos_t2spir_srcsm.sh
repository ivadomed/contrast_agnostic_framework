#!/usr/bin/env bash
# Predict on KIDNEY-T2W with the CHAOS-TRAINED T2spir srcsm model model.
# KIDNEY-T2W has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_15_predict_chaos_t2spir_srcsm.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_srcsm_20260709_121945}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
