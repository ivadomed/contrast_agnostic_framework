#!/usr/bin/env bash
# Train OURS (auglabAug_v26_6_2, train050, DualVal trainer) on autopet PET — ONE training
# run producing BOTH the val000 and val100 checkpoint mirrors. 3 folds, 1 GPU/fold, 2000
# epochs. RUN_ID must contain "_val000_" exactly once (see the CT wrapper's note).
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAutoPETAugLabDualVal"
DATASET_ID="121"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_auglabAug_v26_6_2_train050_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
