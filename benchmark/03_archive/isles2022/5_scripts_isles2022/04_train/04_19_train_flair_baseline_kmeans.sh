#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (isles2022 FLAIR), rung 2. See 04_16 for the dwi
# version / full rationale. 3 folds, 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env_flair.sh"

METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerISLES2022AugLabDefault"
DATASET_ID="${DATASET_ID_FLAIR}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_isles2022_flair_baseline_kmeans"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
