#!/usr/bin/env bash
# Train the BraTS 2024 Glioma T1n-only baseline (Dataset051).
# Standard nnUNet augmentation; no synthesis.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_19_train_t1n_baseline.sh           # auto RUN_ID
#   bash 04_19_train_t1n_baseline.sh brats2024-glioma_t1n_baseline_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT1nBaseline"
DATASET_ID="051"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t1n_baseline"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"
source "$(dirname "$0")/04_00_common.sh" "$@"
