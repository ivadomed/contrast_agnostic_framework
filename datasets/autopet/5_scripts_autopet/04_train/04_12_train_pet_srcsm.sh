#!/usr/bin/env bash
# Train srcsm on autopet PET — AugLab-category. 3 folds, 1 GPU/fold, 2000 epochs.
# MEASURE per-epoch cost on THIS dataset/modality before packing on TamIA (see the CT
# wrapper's note — do not assume the CT-measured ratio transfers to PET either; PET's
# SUV intensity distribution and noise profile differ from CT's HU scale).
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="121"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_srcsm"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
