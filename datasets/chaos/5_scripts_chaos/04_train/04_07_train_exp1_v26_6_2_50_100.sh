#!/usr/bin/env bash
# exp1: V26_6_2 on CHAOS, fold 0 only, 50% train synth / 100% val synth, 300 epochs.
# Runs on set_slot 1 / GPU 1.
#
# Usage:
#   bash 04_07_train_exp1_v26_6_2_50_100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2_50_100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-300}"

export SINGLE_FOLD="${SINGLE_FOLD:-0}"
export SINGLE_SLOT="${SINGLE_SLOT:-1}"
export SINGLE_GPU="${SINGLE_GPU:-1}"

source "$(dirname "$0")/04_00_common.sh" "$@"
