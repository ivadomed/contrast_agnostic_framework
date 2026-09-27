#!/usr/bin/env bash
# Train the BraTS 2024 Glioma T1c-only baseline (Dataset054).
# Standard nnUNet augmentation; no synthesis.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_70_train_t1c_baseline.sh           # auto RUN_ID
#   bash 04_70_train_t1c_baseline.sh brats2024-glioma_t1c_baseline_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env_t1c.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cBaseline"
DATASET_ID="054"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t1c_baseline"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"
# nnUNet-category run: pin results to the contrast tree (matches predict/eval; on TamIA tamia_env.sh otherwise
# points nnUNet_results at a flat scratch dir and the run must be moved by hand afterwards — found on the T2f port).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"

source "$(dirname "$0")/04_00_common.sh" "$@"
