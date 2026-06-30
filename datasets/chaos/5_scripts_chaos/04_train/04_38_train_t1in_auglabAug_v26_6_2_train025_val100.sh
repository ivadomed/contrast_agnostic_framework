#!/usr/bin/env bash
# AugLab augs + V26_6_2 GPU transform on CHAOS MR T1in: 25% train synth / 100% val synth.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_38_train_t1in_auglabAug_v26_6_2_train025_val100.sh                                              # auto RUN_ID
#   bash 04_38_train_t1in_auglabAug_v26_6_2_train025_val100.sh chaos_t1in_auglabAug_v26_6_2_train025_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t1in_auglabAug_v26_6_2_train025_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
