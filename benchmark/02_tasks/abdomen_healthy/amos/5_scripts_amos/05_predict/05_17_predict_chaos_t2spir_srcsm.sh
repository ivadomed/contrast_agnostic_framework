#!/usr/bin/env bash
# Predict on AMOS CT+MRI with the CHAOS T2spir srcsm model (SRCSM SemRandConv-3D,
# Thaler et al. 2025 — 7th ablation arm). Uses chaos Dataset061_CHAOS_MR_T2spir
# checkpoints (pre-exports override env.sh defaults). srcsm reuses the AugLab-default
# trainer (auglab category).
# Usage: bash 05_17_predict_chaos_t2spir_srcsm.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_srcsm_20260709_121945}"
source "$(dirname "$0")/05_01_predict_chaos_common.sh" "$RUN_ID" "${@:2}"
