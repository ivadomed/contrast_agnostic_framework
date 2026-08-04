#!/usr/bin/env bash
# Train the atlas-liver-hcc real-data BASELINE (no synthesis, standard nnUNet
# augmentation) on T1w CE-MRI. Reference point for the augmentation methods'
# generalization to the held-out test patients. 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_01_train_baseline.sh                              # auto RUN_ID: atlas-liver-hcc_t1w_baseline_<TS>
#   bash 04_01_train_baseline.sh atlas-liver-hcc_t1w_baseline_<TS>   # resume an existing RUN_ID
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerAtlasLiverHCCBaseline"
DATASET_ID="080"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

# baseline is an nnUNet-category method -> uses env.sh's nnUNet_results default
# (.../01_predictions/atlas_liver_hcc_model/t1w/nnUNet). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
