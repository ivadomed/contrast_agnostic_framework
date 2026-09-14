#!/usr/bin/env bash
# Train the autopet real-data BASELINE (no synthesis, standard nnUNet augmentation) on
# CT. Reference point for the augmentation methods' cross-contrast (-> PET) and
# cross-institution (-> PSMA cohort) generalization. 3 folds (0 1 2), 1 GPU/fold, 2000
# epochs.
#
# Usage:
#   bash 04_01_train_ct_baseline.sh                            # auto RUN_ID
#   bash 04_01_train_ct_baseline.sh autopet_ct_baseline_<TS>    # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerAutoPETBaseline"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

# baseline is an nnUNet-category method -> uses env.sh's nnUNet_results default
# (.../autopet_model/ct/nnUNet). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
