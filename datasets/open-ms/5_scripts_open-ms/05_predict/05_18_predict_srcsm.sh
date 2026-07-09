#!/usr/bin/env bash
# Predict with srcsm (SRCSM SemRandConv-3D, added 7th comparison arm) on the open-ms test set.
# Usage: bash 05_18_predict_srcsm.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_18_predict_srcsm.sh open-ms_srcsm_<TS> all
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
