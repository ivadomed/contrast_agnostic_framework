#!/usr/bin/env bash
# Train V26_6_2 on CHAOS MR T1in: 50% train synth / 100% val synth.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_34_train_t1in_v26_6_2_train050_val100.sh                                        # auto RUN_ID
#   bash 04_34_train_t1in_v26_6_2_train050_val100.sh chaos_t1in_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t1in_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerCHAOSV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
source "$(dirname "$0")/04_00_common.sh" "$@"
