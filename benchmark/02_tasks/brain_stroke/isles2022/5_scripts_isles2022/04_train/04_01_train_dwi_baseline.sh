#!/usr/bin/env bash
# Train the isles2022 real-data BASELINE (no synthesis, standard nnUNet augmentation) on
# DWI (early post-contrast DCE phase). Reference point for the augmentation
# methods' generalization to the held-out test patients (scored on BOTH dwi and
# flair — see 06_evaluate/06_01_evaluate_run.sh). 3 folds (0 1 2), 1 GPU/fold, 1000
# epochs (see 04_00_common.sh's EPOCH POLICY comment — deliberately half ambl's
# 2000, since ambl's own overfitting on 63 cases/fold is why isles2022 exists).
#
# Usage:
#   bash 04_01_train_dwi_baseline.sh                            # auto RUN_ID: isles2022_dwi_baseline_<TS>
#   bash 04_01_train_dwi_baseline.sh isles2022_dwi_baseline_<TS>   # resume an existing RUN_ID
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerISLES2022Baseline"
DATASET_ID="${DATASET_ID_DWI}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_isles2022_dwi_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

# baseline is an nnUNet-category method -> uses env.sh's nnUNet_results default
# (.../isles2022_model/dwi/nnUNet). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
