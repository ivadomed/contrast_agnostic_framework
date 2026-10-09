#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (on-harmony dwi_ap), rung 2 -- K-means intensity
# clustering only (no label remap, no Voronoi sub-parcellation). Shared,
# dataset-agnostic AugLab config already used by chaos's/brats2024-glioma's/
# on-harmony T1w/T2w's equivalent rung. Usage: bash 04_43_train_dwi_ap_baseline_kmeans.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_dwi_ap_baseline_kmeans"
export DATASET_ID="033"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
