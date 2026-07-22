#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @50% (train), and REAL synth-only
# validation @100% (val), on CHAOS MR T2spir. Closes the matched train050
# val000/val100 pair (val000 sibling: 04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh)
# for the deployed-config ladder — mirrors T1in's 04_16 train050_val100 counterpart.
#
# Train pipeline: full AugLab augs + V26 synth at prob 0.5 (train050 config).
# Val pipeline:   synth-only, V26 synth at prob 1.0 (VALsynthonly config) — applied
#                 in a custom validation_step (real val synth, affects metrics).
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_58_train_t2spir_auglabAug_v26_6_2_train050_val100.sh                                           # auto RUN_ID
#   bash 04_58_train_t2spir_auglabAug_v26_6_2_train050_val100.sh chaos_t2spir_auglabAug_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_auglabAug_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
