#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @50% (train), and REAL synth-only
# validation @100% (val). Folds 0,1 → GPUs 0,1. 200 epochs.
#
# Train pipeline: full AugLab augs + V26 synth at prob 0.5 (train050 config).
# Val pipeline:   synth-only, V26 synth at prob 1.0 (VALsynthonly config) — applied
#                 in a custom validation_step (real val synth, affects metrics).
#
# Usage:
#   bash 04_16_train_auglabAug_v26_6_2_train050_val100.sh [RUN_ID]

# FOLD_SLOT_GPU="2,0,0 3,1,1" \
#   bash datasets/chaos/5_scripts_chaos/04_train/04_16_train_auglabAug_v26_6_2_train050_val100.sh \
#   auglabAug_v26_6_2_train050_val100_20260616_112420


source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

# folds 0,1 → one per GPU (0,1)
export FOLD_SLOT_GPU="${FOLD_SLOT_GPU:-0,0,0 1,1,1}"

source "$(dirname "$0")/04_00_common.sh" "$@"
