#!/usr/bin/env bash
# Evaluate one open-ms T1w experiment against ground truth (Dice + HD95, per contrast).
# Thin wrapper around 06_01_evaluate_run.sh that activates the T1w env
# (TRAINING_CONTRAST=t1w, Dataset071 GT, correct PREDICTIONS_ROOT/METRICS_ROOT).
# Mirrors CHAOS's 06_07_evaluate_t2spir.sh.
#
# Usage: bash 06_13_evaluate_t1w.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
# Example: bash 06_13_evaluate_t1w.sh open-ms_t1w_baseline_20260708_083441 nnUNet
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
export DATASET_ID="71"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "${HERE}/06_01_evaluate_run.sh" "$@"
