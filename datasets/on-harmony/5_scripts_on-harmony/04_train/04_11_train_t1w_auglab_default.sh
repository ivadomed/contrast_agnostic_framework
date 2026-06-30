#!/usr/bin/env bash
# Train AugLab default GPU augmentation on ON-Harmony T1w, 2000 epochs.
# Standard spatial + intensity augmentations, validation on clean data.
# Usage: bash 04_11_train_t1w_auglab_default.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_auglab_default"
export NNUNET_NUM_EPOCHS=2000

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
