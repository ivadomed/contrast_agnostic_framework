#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (BraTS T2w), rung 4 -- full PALETTE partition
# (K-means + label-remap + Voronoi sub-parcellation) but fill is NOISE, not real
# intensity remap (label_fill_noise=true). Mirrors 04_42 (the T1n sibling)
# exactly, just on T2w. This is the noise-fill half of the rung 4->5 causal
# isolation (rung 5 = v26_6_2 alone, 04_14, IDENTICAL partition, real-intensity
# fill) -- the one-variable texture test.
#
# Usage: bash 04_46_train_t2w_baseline_kmeans_label_remap_voronoi.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_t2w_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
