#!/usr/bin/env bash
# Train auglab_default (full AugLab augmentation, NO synthesis) on autopet CT — the
# augmentation-without-synthesis reference. 3 folds, 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_default"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_auglab_default"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
