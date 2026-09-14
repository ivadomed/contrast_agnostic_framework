#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (ispy2 T1WCE), rung 3 -- + label remap on top of
# K-means clustering. 3 folds (0 1 2), 1 GPU/fold, 1000 epochs.
#
# Usage: bash 04_17_train_t1wce_baseline_kmeans_label_remap.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
DATASET_ID="${DATASET_ID_T1WCE}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t1wce_baseline_kmeans_label_remap"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
