#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, new rung between baseline (floor) and
# baseline_kmeans_label_remap, @50% train, 0% val synth (clean validation) —
# same K-means-only config as 04_29, but AugLabDefault trainer instead of
# AugLabValSynth so validation runs on real images only (matches how 04_24/25/26
# cover the rest of this ladder at train050_val000).
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category.
#
# Usage:
#   bash 04_30_train_baseline_kmeans_train050_val000.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline_kmeans_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"

# AugLabDefault trainer -> clean (real-image) validation, no AUGLAB_VAL_PARAMS_GPU_JSON needed.
# nnUNet-category model -> uses env.sh's nnUNet_results default (.../flair/nnUNet).
# No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
