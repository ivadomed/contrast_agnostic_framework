#!/usr/bin/env bash
# Predict with the T1n baseline on the held-out BraTS test set, single fold,
# across all (or selected) contrasts.
#
# Usage:
#   bash 05_01_predict_t1n_baseline.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_01_predict_t1n_baseline.sh t1n_baseline_20260606_162001 0

set -euo pipefail
METHOD="t1n_baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT1nBaseline"
source "$(dirname "$0")/05_predict_common.sh" "$@"
