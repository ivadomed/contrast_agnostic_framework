#!/usr/bin/env bash
# V26_6_2 on CHAOS, train synth 25% / val synth 0%, folds 0+1 packed on slot 1 / GPU 1.
# 200 epochs. DA_WORKERS=8 (two folds share one slot's CPUs).
#
# Usage:
#   bash 04_12_train_v26_6_2_train025_val000.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train025_val000"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train025_val000"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-8}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2_train025_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

export FOLD_SLOT_GPU="${FOLD_SLOT_GPU:-0,1,1 1,1,1}"

source "$(dirname "$0")/04_00_common.sh" "$@"
