#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (T2spir), rung 1 — K-means intensity
# clustering + noise fill ONLY, no per-anatomical-label remap
# (label_remap_prob=0.0, no Voronoi, skip_sub_parc_prob=1.0). Same shared
# AugLab config already used by the T1in ladder's equivalent rung
# (dataset-agnostic — label_classes auto-detected).
#
# Uses nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# instead of training this config twice — see that trainer's module docstring
# (datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py). The
# v26_6_2 headline (rung 4) already exists for T2spir
# (chaos_t2spir_v26_6_2_train050_val100_20260620_112122) — not retrained here.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_55_train_t2spir_baseline_kmeans.sh                                        # auto RUN_ID
#   bash 04_55_train_t2spir_baseline_kmeans.sh chaos_t2spir_baseline_kmeans_train050_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="baseline_kmeans_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t2spir_baseline_kmeans_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans.json"

# nnUNet-category model -> uses env_t2spir.sh's nnUNet_results default. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
