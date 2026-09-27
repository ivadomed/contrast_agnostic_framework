#!/usr/bin/env bash
# Train or resume the BraTS 2024 Glioma T1n-only baseline.
# Usage: bash 04_02_train_t1n_baseline.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="t1n_baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT1nBaseline"
DATASET_ID="051"
DA_WORKERS=64
LOG_DIR="/tmp/nnunet_brats2024_t1n_baseline"
source "$(dirname "$0")/04_00_common.sh" "$@"
