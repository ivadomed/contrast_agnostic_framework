#!/usr/bin/env bash
# Train baseline nnUNet on CHAOS MR T2spir (real data, no synthesis).
# 4 folds, 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_27_train_t2spir_baseline.sh                        # auto RUN_ID
#   bash 04_27_train_t2spir_baseline.sh chaos_t2spir_baseline_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_baseline"
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"
source "$(dirname "$0")/04_00_common.sh" "$@"
