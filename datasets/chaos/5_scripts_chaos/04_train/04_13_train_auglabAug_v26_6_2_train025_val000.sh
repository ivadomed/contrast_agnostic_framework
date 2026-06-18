#!/usr/bin/env bash
# AugLab default augmentations + V26_6_2 GPU transform at 25% (train), val synth 0%
# (default AugLab trainer → no val augmentation). Folds 0+1 packed on slot 3 / GPU 3.
# 200 epochs. Default auglab augmentations kept intact; only the V26_6_2 synth
# transform probability is 0.25 (see config _train025.json).
#
# Usage:
#   bash 04_13_train_auglabAug_v26_6_2_train025_val000.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train025_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_train025_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

export FOLD_SLOT_GPU="${FOLD_SLOT_GPU:-0,3,3 1,3,3}"

source "$(dirname "$0")/04_00_common.sh" "$@"
