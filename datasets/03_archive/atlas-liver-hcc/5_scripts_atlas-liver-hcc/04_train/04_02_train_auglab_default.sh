#!/usr/bin/env bash
# Train auglab_default (full AugLab augmentation, NO synthesis) on atlas-liver-hcc T1w —
# the augmentation-without-synthesis reference. 3 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_02_train_auglab_default.sh                          # auto RUN_ID
#   bash 04_02_train_auglab_default.sh atlas-liver-hcc_t1w_auglab_default_<TS>   # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_default"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="080"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_auglab_default"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
