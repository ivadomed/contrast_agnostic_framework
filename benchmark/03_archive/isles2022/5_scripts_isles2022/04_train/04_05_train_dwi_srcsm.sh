#!/usr/bin/env bash
# Train srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) on isles2022 DWI — an added
# comparison arm (AugLab-category, reuses the auglab_default trainer).
# 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# NOTE: measure the srcsm-vs-others per-epoch cost ratio on THIS dataset before
# assuming ambl's figure transfers (CLAUDE.md TamIA section: brats/on-harmony saw
# ~3x, atlas-liver-hcc saw only ~1.2x, ambl saw ~1.5x, a since-abandoned dataset saw
# no slowdown at all) — relevant when packing folds on TamIA later.
#
# Usage:
#   bash 04_05_train_dwi_srcsm.sh                          # auto RUN_ID
#   bash 04_05_train_dwi_srcsm.sh isles2022_dwi_srcsm_<TS>   # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerISLES2022AugLabDefault"
DATASET_ID="${DATASET_ID_DWI}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_isles2022_dwi_srcsm"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
