#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (open-ms T1w), rung 4 -- full PALETTE partition
# (K-means + label-remap + Voronoi sub-parcellation) but fill is NOISE, not real
# intensity remap, @50% train, 0% val synth (clean validation). Mirrors 04_25
# (the FLAIR sibling) exactly, just on T1w. This is the noise-fill half of the
# rung 4->5 causal isolation (rung 5 = v26_6_2 alone, IDENTICAL partition,
# real-intensity fill) -- the one-variable texture test. 3 folds (0,1,2),
# 1 GPU/fold, 2000 epochs. nnUNet-category.
#
# Usage:
#   bash 04_37_train_t1w_baseline_kmeans_label_remap_voronoi_train050_val000.sh
source "$(dirname "$0")/../00_utils/env_t1w.sh"

METHOD="baseline_kmeans_label_remap_voronoi_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="071"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_t1w_baseline_kmeans_label_remap_voronoi_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"

# AugLabDefault trainer -> clean (real-image) validation, no AUGLAB_VAL_PARAMS_GPU_JSON needed.
# nnUNet-category model -> uses env.sh's nnUNet_results default (.../t1w/nnUNet).
# No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
