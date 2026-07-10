#!/usr/bin/env bash
# AugLab default augmentations + V26_6_2 GPU transform @50% (train), val synth 0%, on
# CHAOS MR T2spir. Train synth = config prob 0.5; validation runs on clean data (stock
# nnUNet validation_step) — mirrors 04_09's T1in train050_val000 counterpart.
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh                                           # auto RUN_ID
#   bash 04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh chaos_t2spir_auglabAug_v26_6_2_train050_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_auglabAug_v26_6_2_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
