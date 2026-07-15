#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (T2spir), rung 2 — K-means + per-anatomical-
# label remap, no Voronoi (skip_sub_parc_prob=1.0). Same shared AugLab config
# already used by the T1in ladder's equivalent rung (dataset-agnostic).
#
# Uses nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# See datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_56_train_t2spir_baseline_kmeans_label_remap.sh
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="baseline_kmeans_label_remap_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t2spir_baseline_kmeans_label_remap_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans_label_remap.json"

# nnUNet-category model -> uses env_t2spir.sh's nnUNet_results default. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
