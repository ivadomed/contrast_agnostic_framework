#!/usr/bin/env bash
# Train srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) on toothfairy2 CBCT —
# AugLab-category, reuses the auglab_default trainer. 3 folds, 1 GPU/fold, 1000 ep.
#
# ⚠️ MEASURE srcsm's per-epoch cost on THIS dataset before packing folds on TamIA.
# CLAUDE.md is explicit that the ~3x slowdown seen on brats/on-harmony does NOT
# transfer (atlas-liver-hcc ~1.2x, ambl ~1.5x, one dataset showed none at all). If
# srcsm IS the straggler here, give it its own GPU via PACK_GPU_MAP rather than
# letting round-robin pair it with a second fold and hold the whole node open.
#
# Usage: bash 04_05_train_srcsm.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
DATASET_ID="${DATASET_ID_CBCT}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_toothfairy2_cbct_srcsm"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
