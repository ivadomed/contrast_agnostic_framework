#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T2f/FLAIR), rung 2 -- K-means intensity
# clustering + noise fill ONLY, no per-label remap, no Voronoi sub-parcellation
# (label_remap_prob=0.0, skip_sub_parc_prob=1.0). Same shared, dataset-agnostic
# AugLab config already used by T1n's/T2w's equivalent rung.
# Uses Dataset053_BraTS2024GliomaT2f (single FLAIR channel).
#
# Usage: bash 04_57_train_t2f_baseline_kmeans.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env_t2f.sh"

METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fAugLabDefault"
DATASET_ID="053"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t2f_baseline_kmeans"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"

DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
