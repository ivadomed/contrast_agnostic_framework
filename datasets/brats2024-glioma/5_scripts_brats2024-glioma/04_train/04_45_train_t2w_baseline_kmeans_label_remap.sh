#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T2w), rung 3 -- + label remap on top of
# K-means clustering. Mirrors 04_41 (the T1n sibling) exactly, just on T2w.
#
# Usage: bash 04_45_train_t2w_baseline_kmeans_label_remap.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t2w_baseline_kmeans_label_remap"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
