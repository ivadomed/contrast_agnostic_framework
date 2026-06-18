#!/usr/bin/env bash
# Train V26_6_2 (train_synth_prob=0.50, val_synth_prob=1.0) on BraTS 2024 Glioma.
# Usage: bash 04_10_train_v26_6_2_train050_val100.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100"
DATASET_ID="051"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_v26_6_2_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1500}"
source "$(dirname "$0")/04_00_common.sh" "$@"
