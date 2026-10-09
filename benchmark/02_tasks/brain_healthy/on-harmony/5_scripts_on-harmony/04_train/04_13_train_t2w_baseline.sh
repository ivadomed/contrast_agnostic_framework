#!/usr/bin/env bash
# Train baseline (no synthesis) on ON-Harmony T2w, 2000 epochs, 4 folds.
# Usage: bash 04_13_train_t2w_baseline.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
DA_WORKERS=64
LOG_DIR="/tmp/nnunet_on-harmony_t2w_baseline"
export DATASET_ID="032"
export NNUNET_NUM_EPOCHS=2000
source "$(dirname "$0")/04_00_common.sh" "$@"
