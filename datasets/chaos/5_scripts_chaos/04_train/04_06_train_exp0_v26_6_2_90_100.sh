#!/usr/bin/env bash
# exp0: V26_6_2 on CHAOS, fold 0 only, default synth prob (90% train / 100% val), 300 epochs.
# Runs on GPU 0 (SINGLE_GPU=0).
#
# Usage:
#   bash 04_06_train_exp0_v26_6_2_90_100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train090_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2_90_100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-300}"

export SINGLE_FOLD=0
export SINGLE_SLOT=0
export SINGLE_GPU=0

source "$(dirname "$0")/04_00_common.sh" "$@"
