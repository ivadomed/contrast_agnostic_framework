#!/usr/bin/env bash
# Train baseline (no synthesis) on ON-Harmony dwi_ap, 2000 epochs, 4 folds.
# Usage: bash 04_36_train_dwi_ap_baseline.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_on-harmony_dwi_ap_baseline"
export DATASET_ID="033"
export NNUNET_NUM_EPOCHS=2000
# nnUNet-category run: set the results base EXPLICITLY. 04_00_common.sh re-sources env.sh, which
# unconditionally re-exports nnUNet_results to the repo path and would clobber a cluster override
# (on TamIA that lands checkpoints in /project — 498K/500K files — instead of $SCRATCH).
# PREDICTIONS_ROOT is set conditionally in common_env, so it survives; on Vulcan this resolves to
# the identical path env.sh would have used.
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"

source "$(dirname "$0")/04_00_common.sh" "$@"
