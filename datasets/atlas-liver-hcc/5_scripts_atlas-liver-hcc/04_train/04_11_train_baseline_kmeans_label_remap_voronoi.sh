#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (atlas-liver-hcc T1w), rung 4 -- full PALETTE
# partition (K-means + label-remap + Voronoi sub-parcellation) but fill is NOISE,
# not real intensity remap (label_fill_noise=true). Shared, dataset-agnostic AugLab
# config already used by chaos's/brats2024-glioma's/on-harmony's/open-ms's equivalent
# rung. This is the noise-fill half of the rung 4->5 causal isolation (rung 5 =
# v26_6_2 alone, 04_12, IDENTICAL partition, real-intensity fill) -- the
# one-variable texture test. 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage: bash 04_11_train_baseline_kmeans_label_remap_voronoi.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="080"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
