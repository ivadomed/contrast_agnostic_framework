#!/usr/bin/env bash
# Train AugLab default GPU augmentation on ON-Harmony T2w, 2000 epochs.
# Standard spatial + intensity augmentations, validation on clean data.
# Usage: bash 04_17_train_t2w_auglab_default.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t2w_auglab_default"
export DATASET_ID="032"
export NNUNET_NUM_EPOCHS=2000

# Must export explicitly: the shared train driver exports AUGLAB_PARAMS_GPU_JSON as an
# empty string when unset, which defeats the trainer's "unset → default config" fallback
# (empty ≠ unset) → open('') crash. Matches the sibling auglab scripts (04_15/16/18).
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_default01-23.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
