#!/usr/bin/env bash
# Train or resume the BraTS 2024 Glioma baseline (vanilla nnUNet, no synthesis).
# Usage: bash 04_01_train_baseline.sh [RUN_ID]   (omit RUN_ID for fresh run)

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaBaseline"
DA_WORKERS=64
LOG_DIR="/tmp/nnunet_brats2024_baseline"
source "$(dirname "$0")/04_00_common.sh" "$@"
