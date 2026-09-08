#!/usr/bin/env bash
# Train OURS (auglabAug_v26_6_2, train050, DualVal trainer) on toothfairy2 CBCT —
# ONE training run producing BOTH the val000 and val100 checkpoint mirrors (see
# toothfairy2/trainers/auglab_dualval.py). 3 folds, 1 GPU/fold, 1000 epochs.
#
# ⚠️ The RUN_ID passed here MUST contain "_val000_" exactly once — the DualVal
# trainer's on_train_end() materializes the "_val100_" sibling directory from it.
# train_common.sh's auto-generated RUN_ID already does this if METHOD is left as
# below; if you pass an explicit RUN_ID, keep the "_val000_" marker.
#
# Usage: bash 04_06_train_auglabAug_v26_6_2_dualval.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDualVal"
DATASET_ID="${DATASET_ID_CBCT}"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_toothfairy2_cbct_auglabAug_v26_6_2_train050_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
