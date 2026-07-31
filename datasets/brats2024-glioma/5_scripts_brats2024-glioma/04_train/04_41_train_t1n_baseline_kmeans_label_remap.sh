#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T1n), rung 3 -- K-means clustering +
# per-anatomical-label remap, no Voronoi sub-parcellation (label_remap_prob=0.5,
# skip_sub_parc_prob=1.0). Same shared, dataset-agnostic AugLab config already
# used by open-ms's and CHAOS's equivalent rung.
#
# Usage: bash 04_41_train_t1n_baseline_kmeans_label_remap.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t1n_baseline_kmeans_label_remap"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"

DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
