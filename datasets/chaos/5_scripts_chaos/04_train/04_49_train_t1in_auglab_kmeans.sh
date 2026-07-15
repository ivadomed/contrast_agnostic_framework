#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, new rung between auglab_default (floor) and
# auglab_kmeans_label_remap — K-means intensity clustering + noise fill ONLY,
# no per-anatomical-label remap (label_remap_prob=0.0, no Voronoi,
# skip_sub_parc_prob=1.0, full AugLab default01-23 suite held fixed). Same
# shared AugLab config already used by open-ms's equivalent rung
# (dataset-agnostic — label_classes auto-detected).
#
# Uses nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# See datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_49_train_t1in_auglab_kmeans_dualval.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_train025_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t1in_auglab_kmeans_train025_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglab_kmeans_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
