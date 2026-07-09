#!/usr/bin/env bash
# Train srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) on open-ms FLAIR — an added 7th
# comparison arm (AugLab-category, reuses the auglab_default trainer). 4 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_20_train_srcsm.sh                          # auto RUN_ID
#   bash 04_20_train_srcsm.sh open-ms_srcsm_<TS>       # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_srcsm"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
