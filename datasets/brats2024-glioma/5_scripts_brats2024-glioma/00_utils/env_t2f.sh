#!/usr/bin/env bash
# T2f (FLAIR) contrast override — source this instead of env.sh for FLAIR experiments.
# Sets TRAINING_CONTRAST and nnUNet_results BEFORE sourcing env.sh so that when
# 04_00_common.sh re-sources env.sh the conditional assignments (${VAR:-default})
# preserve these values rather than overwriting them with the t1n defaults.
#
# TRAINING_CONTRAST is "t2f" (not "flair") to match the column_order/contrast token
# used everywhere else in this dataset's eval configs (column_order: [t1c, t1n, t2f,
# t2w]) — significance_from_config.py's "exclude the training contrast" logic keys off
# this exact string.

export TRAINING_CONTRAST="t2f"
export nnUNet_results="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2f/nnUNet"

# T2f training jobs run for 2500 epochs; the longest auglab experiments take ~90s/epoch
# (~62h total). Default to a ~7-day limit, but allow a caller to override (e.g. a resume
# that only needs 2 days) by exporting RUN_JOB_TIME_DEFAULT before invoking.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-6-23:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
