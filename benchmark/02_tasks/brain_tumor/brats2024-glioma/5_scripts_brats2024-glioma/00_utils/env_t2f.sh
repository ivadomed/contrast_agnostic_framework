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
# Guarded (${nnUNet_results:-...}), matching env.sh's own t1n default — NOT an
# unconditional assignment. On a plain Vulcan/Killarney shell nnUNet_results is
# unset when this file is first sourced, so the guard has no effect and this
# still resolves to the usual t2f path. On TamIA, scripts/cluster/tamia_env.sh
# is sourced AFTER this file (per that file's own contract) and exports its own
# scratch-resident nnUNet_results — but every 04_train wrapper re-sources THIS
# file at its own top, so an unconditional assignment here would silently clobber
# tamia_env.sh's override back to the /project path on every wrapper invocation.
# Hit for real 2026-09-17: a baseline + v26_6_2_train050_val100 TamIA pack wrote
# checkpoints to /project (tight file-count quota) instead of /scratch before
# being caught and cancelled. env_t2w.sh had the identical unconditional-assignment
# bug (never triggered because t2w baseline/v26_6_2 predate TamIA); fixed there too.
export nnUNet_results="${nnUNet_results:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2f/nnUNet}"

# T2f training jobs run for 2500 epochs; the longest auglab experiments take ~90s/epoch
# (~62h total). Default to a ~7-day limit, but allow a caller to override (e.g. a resume
# that only needs 2 days) by exporting RUN_JOB_TIME_DEFAULT before invoking.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-6-23:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
