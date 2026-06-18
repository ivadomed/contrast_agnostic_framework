#!/usr/bin/env bash
# V26_6_2 on CHAOS: train synth 25% / val synth 100%. All 4 folds, 1 fold/GPU, 200 epochs.
#
# Usage:
#   bash 04_19_train_v26_6_2_train025_val100.sh                                         # auto: chaos_v26_6_2_train025_val100_<TS>
#   bash 04_19_train_v26_6_2_train025_val100.sh chaos_v26_6_2_train025_val100_<TS>      # explicit RUN_ID to resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train025_val100"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-8}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

source "$(dirname "$0")/04_00_common.sh" "$@"
