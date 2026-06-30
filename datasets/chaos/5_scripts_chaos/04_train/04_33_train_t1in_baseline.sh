#!/usr/bin/env bash
# Train baseline nnUNet on CHAOS MR T1in (real data, no synthesis).
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_33_train_t1in_baseline.sh                          # auto RUN_ID
#   bash 04_33_train_t1in_baseline.sh chaos_t1in_baseline_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_t1in_baseline"
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"
source "$(dirname "$0")/04_00_common.sh" "$@"
