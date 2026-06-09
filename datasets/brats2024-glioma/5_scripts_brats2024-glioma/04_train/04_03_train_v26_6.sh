#!/usr/bin/env bash
# Train or resume V26_6 on-the-fly GPU synthesis on BraTS 2024 Glioma (T1n).
# Uses Dataset051_BraTS2024GliomaT1n (single T1n channel).
# Usage: bash 04_03_train_v26_6.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_v26_6"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS=1000
source "$(dirname "$0")/04_00_common.sh" "$@"
