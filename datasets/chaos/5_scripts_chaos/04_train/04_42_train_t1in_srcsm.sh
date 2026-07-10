#!/usr/bin/env bash
# Train SRCSM (SemRandConv-3D, Thaler et al. 2025) GPU augmentation on CHAOS MR T1in.
# Reuses the CHAOS AugLab trainer; selected purely via the srcsm SemRandConv config.
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_42_train_t1in_srcsm.sh                          # auto RUN_ID
#   bash 04_42_train_t1in_srcsm.sh chaos_t1in_srcsm_<TS>    # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t1in_srcsm"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
