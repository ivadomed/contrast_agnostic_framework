#!/usr/bin/env bash
# Predict on AMOS CT+MRI with the CHAOS T2spir SynthSeg-EM model.
# Uses chaos Dataset061_CHAOS_MR_T2spir checkpoints (pre-exports override env.sh defaults).
# Usage: bash 05_11_predict_chaos_t2spir_synthseg_EM.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_synthseg_EM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_synthseg_EM_20260620_112357}"
source "$(dirname "$0")/05_01_predict_chaos_common.sh" "$RUN_ID" "${@:2}"
