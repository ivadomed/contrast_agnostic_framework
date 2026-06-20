#!/usr/bin/env bash
# Train V26_6_2 on CHAOS MR T2spir: 50% train synth / 100% val synth.
# 4 folds, 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_28_train_t2spir_v26_6_2_train050_val100.sh                                     # auto RUN_ID
#   bash 04_28_train_t2spir_v26_6_2_train050_val100.sh chaos_t2spir_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_v26_6_2_train050_val100"
# torch.compile requires Triton which is not available on Vulcan — keep disabled.
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"
source "$(dirname "$0")/04_00_common.sh" "$@"
