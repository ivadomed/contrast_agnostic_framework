#!/usr/bin/env bash
# Evaluate one picai-prostate ADC experiment against ground truth (Dice + HD95, per contrast).
# Thin wrapper around 06_01_evaluate_run.sh that activates the ADC env
# (TRAINING_CONTRAST=adc, Dataset081 GT, correct PREDICTIONS_ROOT/METRICS_ROOT).
# Mirrors open-ms's 06_13_evaluate_t1w.sh / CHAOS's 06_07_evaluate_t2spir.sh.
#
# Usage: bash 06_02_evaluate_adc.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_adc.sh"
export DATASET_ID="81"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "${HERE}/06_01_evaluate_run.sh" "$@"
