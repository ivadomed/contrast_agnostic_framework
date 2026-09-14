#!/usr/bin/env bash
# Train SynthSeg-noEM on autopet PET. 3 folds, 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="121"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_synthseg_noEM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
