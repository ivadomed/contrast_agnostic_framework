#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Train or resume baseline (no synthesis).
# Usage: bash 03_train_baseline.sh [RUN_ID]   (omit RUN_ID for fresh run)
METHOD="baseline"
TRAINER="nnUNetTrainerOnHarmonyBaseline"
DA_WORKERS=64
LOG_DIR="/tmp/nnunet_baseline"
source "$(dirname "$0")/04_00_common.sh" "$@"
