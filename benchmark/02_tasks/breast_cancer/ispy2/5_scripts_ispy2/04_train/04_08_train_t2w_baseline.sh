#!/usr/bin/env bash
# Train the ispy2 real-data BASELINE on T2W. 3 folds, 1 GPU/fold, 1000 epochs.
# See 04_01_train_t1wce_baseline.sh for the full rationale (same method, other contrast).
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerISPY2Baseline"
DATASET_ID="${DATASET_ID_T2W}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t2w_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

source "$(dirname "$0")/04_00_common.sh" "$@"
