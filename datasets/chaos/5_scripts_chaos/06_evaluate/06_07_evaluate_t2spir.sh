#!/usr/bin/env bash
# Evaluate one CHAOS T2spir experiment against ground truth.
# Thin wrapper around 06_01_evaluate_run.sh that activates the T2spir env
# (TRAINING_CONTRAST=t2spir, Dataset061, correct PREDICTIONS_ROOT and METRICS_ROOT).
#
# Usage: bash 06_07_evaluate_t2spir.sh <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default: all)
#
# Example:
#   bash 06_07_evaluate_t2spir.sh chaos_t2spir_baseline_20260620_111146 all

set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
export DATASET_ID="61"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "${HERE}/06_01_evaluate_run.sh" "$@"
