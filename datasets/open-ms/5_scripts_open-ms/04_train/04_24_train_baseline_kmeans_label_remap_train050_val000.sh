#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, rung 2 of 4, @50% train, 0% val synth (clean
# validation) — same K-means intensity clustering + noise fill, no Voronoi
# (skip_sub_parc_prob=1.0), as 04_10, but AugLabDefault trainer instead of
# AugLabValSynth so validation runs on real images only. Companion to
# 04_26 (v26_6_2 alone @ train050_val000) and 04_25 (+Voronoi @ train050_val000).
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category (matches 04_10's placement).
#
# Usage:
#   bash 04_24_train_baseline_kmeans_label_remap_train050_val000.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline_kmeans_label_remap_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"

# AugLabDefault trainer -> clean (real-image) validation, no AUGLAB_VAL_PARAMS_GPU_JSON needed.
# nnUNet-category model -> uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_10. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
