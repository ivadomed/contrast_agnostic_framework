#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Train or resume V26_6 on-the-fly GPU synthesis.
# Usage: bash 03_train_v26_6.sh [RUN_ID]
METHOD="v26_6"
TRAINER="nnUNetTrainerOnHarmonyV26_6"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_v26_6"
source "$(dirname "$0")/04_00_common.sh" "$@"
