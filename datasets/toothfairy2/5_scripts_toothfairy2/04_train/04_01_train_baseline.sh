#!/usr/bin/env bash
# Train the toothfairy2 real-data BASELINE (no synthesis, standard nnUNet
# augmentation) on CBCT. The reference point the augmentation methods' generalization
# is measured against — both on the held-out CBCT test cases and, more importantly,
# on the cross-modality external test sets (hanseg CT + MR-T1, mandible only), where
# a non-contrast-agnostic baseline is expected to fall apart.
# 3 folds (0 1 2), 1 GPU/fold, 1000 epochs.
#
# Usage:
#   bash 04_01_train_baseline.sh                                # auto RUN_ID
#   bash 04_01_train_baseline.sh toothfairy2_cbct_baseline_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerToothFairy2Baseline"
DATASET_ID="${DATASET_ID_CBCT}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_toothfairy2_cbct_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

# nnUNet-category method -> uses env.sh's nnUNet_results default
# (.../toothfairy2_model/cbct/nnUNet). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
