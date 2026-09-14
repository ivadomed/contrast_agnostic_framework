#!/usr/bin/env bash
# Train OURS (auglabAug_v26_6_2, train050, DualVal trainer) on ispy2 T2W — see
# 04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh for the full DualVal rationale.
# 3 folds, 1 GPU/fold, 1000 epochs.
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerISPY2AugLabDualVal"
DATASET_ID="${DATASET_ID_T2W}"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t2w_auglabAug_v26_6_2_train050_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
