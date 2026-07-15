#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, top rung (auglabAug_v26_6_2 — OURS). Same
# shared config already used by the existing auglabAug_v26_6_2_train025_val100
# headline run (dataset-agnostic).
#
# Retrained here (rather than reusing the existing
# chaos_t1in_auglabAug_v26_6_2_train025_val100 run) so it uses
# nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# instead of training this config twice. See
# datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py. The existing
# train025_val100-only run stays valid for the separate 6-method headline table.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_52_train_t1in_auglabAug_v26_6_2_dualval.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train025_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t1in_auglabAug_v26_6_2_train025_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
