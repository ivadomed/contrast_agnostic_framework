#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T2w), rung 2 -- K-means intensity
# clustering only (no label remap, no Voronoi sub-parcellation). Mirrors 04_40
# (the T1n sibling) exactly, just on T2w. Same shared, dataset-agnostic AugLab
# config already used by chaos's/open-ms's equivalent rung.
#
# Usage: bash 04_44_train_t2w_baseline_kmeans.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t2w_baseline_kmeans"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
