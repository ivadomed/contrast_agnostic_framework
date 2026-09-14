#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (autopet CT), rung 2 — K-means intensity clustering
# only (no label remap, no Voronoi sub-parcellation). Shared, dataset-agnostic AugLab
# config, identical to every other dataset's rung 2. 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage: bash 04_16_train_ct_baseline_kmeans.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_baseline_kmeans"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
