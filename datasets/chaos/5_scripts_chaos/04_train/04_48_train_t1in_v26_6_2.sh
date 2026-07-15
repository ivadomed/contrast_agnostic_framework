#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, top rung (v26_6_2 ALONE — real-texture
# fill, the "Ours, v26 only" row). Same shared spatialDA-only config already
# used by the existing v26_6_2_train050_val100 headline run (dataset-agnostic).
#
# Retrained here (rather than reusing the existing chaos_t1in_v26_6_2_train050_val100
# run) so it uses nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# instead of training this config twice. See
# datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py. The existing
# train050_val100-only run stays valid for the separate 6-method headline table.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_48_train_t1in_v26_6_2.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t1in_v26_6_2_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# nnUNet-category model -> uses env.sh's nnUNet_results default. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
