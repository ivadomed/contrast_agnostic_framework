#!/usr/bin/env bash
# Train the open-ms real-data BASELINE (no synthesis, standard nnUNet augmentation) on
# FLAIR. Reference point for the augmentation methods' cross-contrast generalization.
# 4 folds, 1 GPU/fold (each fold = its own Slurm job), 2000 epochs.
#
# Usage:
#   bash 04_01_train_baseline.sh                              # auto RUN_ID: open-ms_baseline_<TS>
#   bash 04_01_train_baseline.sh open-ms_baseline_<TS>        # resume an existing RUN_ID
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerOpenMSBaseline"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

# baseline is an nnUNet-category method → uses env.sh's nnUNet_results default
# (.../01_predictions/open_ms_model/flair/nnUNet). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
