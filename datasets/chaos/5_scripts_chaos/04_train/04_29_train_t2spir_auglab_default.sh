#!/usr/bin/env bash
# Train AugLab default GPU augmentation on CHAOS MR T2spir (standard aug, no synthesis).
# 4 folds, 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_29_train_t2spir_auglab_default.sh                              # auto RUN_ID
#   bash 04_29_train_t2spir_auglab_default.sh chaos_t2spir_auglab_default_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="auglab_default"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_auglab_default"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
