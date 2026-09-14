#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (ispy2 T2W), rung 5 (real-intensity fill — the
# causal pair vs rung 4). See 04_19 for the t1wce version / full rationale.
#
# ⚠️ Same naming trap as 04_19: METHOD is "v26_6_2_train050_val100" — NO
# "auglabAug_" prefix. Always pass a RUN_ID matching METHOD exactly.
#
# 3 folds (0 1 2), 1 GPU/fold, 1000 epochs.
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
DATASET_ID="${DATASET_ID_T2W}"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t2w_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

source "$(dirname "$0")/04_00_common.sh" "$@"
