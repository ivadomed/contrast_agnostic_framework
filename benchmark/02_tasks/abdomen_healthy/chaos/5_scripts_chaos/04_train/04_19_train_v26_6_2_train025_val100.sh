#!/usr/bin/env bash
# V26_6_2 on CHAOS: train synth 25% / val synth 100%. All 4 folds, 1 fold/GPU, 200 epochs.
#
# Usage:
#   bash 04_19_train_v26_6_2_train025_val100.sh                                         # auto: chaos_v26_6_2_train025_val100_<TS>
#   bash 04_19_train_v26_6_2_train025_val100.sh chaos_v26_6_2_train025_val100_<TS>      # explicit RUN_ID to resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train025_val100"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-8}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"


# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerCHAOSV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
source "$(dirname "$0")/04_00_common.sh" "$@"
