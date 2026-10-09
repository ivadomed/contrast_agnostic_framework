#!/usr/bin/env bash
# Train the totalseg-pelvic real-data BASELINE (no synthesis) on CT. Reference point for the augmentation methods' cross-contrast generalization.
# 3 folds (0 1 2), 1 GPU/fold, 200 epochs.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerTotalsegPelvicBaseline"
DATASET_ID="130"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_ct_baseline"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

source "$(dirname "$0")/04_00_common.sh" "$@"
