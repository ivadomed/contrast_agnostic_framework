#!/usr/bin/env bash
# Train OURS (auglabAug_v26_6_2, train050, DualVal trainer) on totalseg-pelvic MRI — ONE run producing BOTH val000 and val100 checkpoint mirrors. RUN_ID MUST contain _val000_ exactly once.
# 3 folds (0 1 2), 1 GPU/fold, 200 epochs.
source "$(dirname "$0")/../00_utils/env_mri.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDualVal"
DATASET_ID="131"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_mri_auglabAug_v26_6_2_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
