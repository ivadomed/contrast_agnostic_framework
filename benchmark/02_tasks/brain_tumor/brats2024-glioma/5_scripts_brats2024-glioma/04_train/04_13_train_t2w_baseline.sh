#!/usr/bin/env bash
# Train the BraTS 2024 Glioma T2w-only baseline (Dataset052).
# Standard nnUNet augmentation; no synthesis.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_13_train_t2w_baseline.sh           # auto RUN_ID
#   bash 04_13_train_t2w_baseline.sh brats2024-glioma_t2w_baseline_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wBaseline"
DATASET_ID="052"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t2w_baseline"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"
source "$(dirname "$0")/04_00_common.sh" "$@"
