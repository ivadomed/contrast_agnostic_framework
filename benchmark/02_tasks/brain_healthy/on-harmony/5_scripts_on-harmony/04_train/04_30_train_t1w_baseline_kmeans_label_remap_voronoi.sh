#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (on-harmony T1w), rung 4 -- full PALETTE partition
# (K-means + label-remap + Voronoi sub-parcellation) but fill is NOISE, not real
# intensity remap (label_fill_noise=true). Shared, dataset-agnostic AugLab config
# already used by chaos's/brats2024-glioma's equivalent rung. This is the
# noise-fill half of the rung 4->5 causal isolation (rung 5 = v26_6_2 alone,
# IDENTICAL partition, real-intensity fill) -- the one-variable texture test.
# Usage: bash 04_30_train_t1w_baseline_kmeans_label_remap_voronoi.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_baseline_kmeans_label_remap_voronoi"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
