#!/usr/bin/env bash
# Predict with V26_6 on the held-out BraTS test set, single fold,
# across all (or selected) contrasts.
#
# Usage:
#   bash 05_02_predict_v26_6.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_02_predict_v26_6.sh v26_6_20260607_001859 0

set -euo pipefail
METHOD="v26_6"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6"
source "$(dirname "$0")/05_predict_common.sh" "$@"
