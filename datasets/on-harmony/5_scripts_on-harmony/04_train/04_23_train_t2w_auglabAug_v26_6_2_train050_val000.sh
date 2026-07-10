#!/usr/bin/env bash
# Train AugLab default augmentations + V26_6_2 synthesis @50% train / 0% val on
# ON-Harmony T2w. Train pipeline: full AugLab augs + V26_6_2 synth at prob 0.5.
# Val pipeline:   clean (un-augmented), stock nnUNet validation_step.
# Usage: bash 04_23_train_t2w_auglabAug_v26_6_2_train050_val000.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t2w_auglabAug_v26_6_2_train050_val000"
export DATASET_ID="032"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
