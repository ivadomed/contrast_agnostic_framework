#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T1n), rung 2 -- K-means intensity
# clustering + noise fill ONLY, no per-label remap, no Voronoi sub-parcellation
# (label_remap_prob=0.0, skip_sub_parc_prob=1.0). Same shared, dataset-agnostic
# AugLab config already used by open-ms's and CHAOS's equivalent rung.
# Uses Dataset051_BraTS2024GliomaT1n (single T1n channel).
#
# Usage: bash 04_40_train_t1n_baseline_kmeans.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t1n_baseline_kmeans"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"

DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
