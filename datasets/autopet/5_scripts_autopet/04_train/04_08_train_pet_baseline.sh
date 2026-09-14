#!/usr/bin/env bash
# Train the autopet real-data BASELINE on PET (FDG, SUV). 3 folds, 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerAutoPETBaseline"
DATASET_ID="121"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

source "$(dirname "$0")/04_00_common.sh" "$@"
