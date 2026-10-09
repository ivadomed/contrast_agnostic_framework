#!/usr/bin/env bash
# Train the isles2022 real-data BASELINE on FLAIR. 3 folds, 1 GPU/fold, 2000 epochs.
# See 04_01_train_dwi_baseline.sh for the full rationale (same method, other contrast).
source "$(dirname "$0")/../00_utils/env_flair.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerISLES2022Baseline"
DATASET_ID="${DATASET_ID_FLAIR}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_isles2022_flair_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

source "$(dirname "$0")/04_00_common.sh" "$@"
