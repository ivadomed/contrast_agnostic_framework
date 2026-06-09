#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Train V26_6_2 on-the-fly GPU synthesis with label-guided intensity remap.
# Usage: bash 04_03_train_v26_6_2.sh [RUN_ID]
METHOD="v26_6_2"
TRAINER="nnUNetTrainerOnHarmonyV26_6_2"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_v26_6_2"
source "$(dirname "$0")/04_00_common.sh" "$@"
