#!/usr/bin/env bash
# Train auglab_default (full AugLab augmentation, NO synthesis) on open-ms T1w — the
# augmentation-without-synthesis reference. 3 folds (0 1 2), 1 GPU/fold, 2000 epochs, 30h.
#
# Usage:
#   bash 04_14_train_t1w_auglab_default.sh                            # auto RUN_ID
#   bash 04_14_train_t1w_auglab_default.sh open-ms_t1w_auglab_default_<TS>   # resume
source "$(dirname "$0")/../00_utils/env_t1w.sh"

METHOD="auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="071"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_t1w_auglab_default"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
