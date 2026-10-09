#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (on-harmony T2w), rung 3 -- + label remap on top
# of K-means clustering. Shared, dataset-agnostic AugLab config already used by
# chaos's/brats2024-glioma's/on-harmony T1w's equivalent rung.
# Usage: bash 04_33_train_t2w_baseline_kmeans_label_remap.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t2w_baseline_kmeans_label_remap"
export DATASET_ID="032"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
