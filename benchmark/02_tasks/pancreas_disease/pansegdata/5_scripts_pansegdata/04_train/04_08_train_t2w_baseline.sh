#!/usr/bin/env bash
# Train the pansegdata real-data BASELINE on T2W. 3 folds, 1 GPU/fold, 2000 epochs.
# See 04_01_train_t1wce_baseline.sh for the full rationale (same method, other contrast).
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerPANSEGDATABaseline"
DATASET_ID="${DATASET_ID_T2W}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_pansegdata_t2w_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

source "$(dirname "$0")/04_00_common.sh" "$@"
