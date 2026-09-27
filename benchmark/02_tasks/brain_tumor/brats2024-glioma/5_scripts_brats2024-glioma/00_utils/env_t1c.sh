#!/usr/bin/env bash
# T1c (contrast-enhanced T1, BIDS "ce-gadolinium_T1w") contrast override — source this instead of
# env.sh for T1c experiments. Sets TRAINING_CONTRAST and nnUNet_results BEFORE sourcing env.sh.
#
# nnUNet_results is GUARDED (${nnUNet_results:-...}) on purpose, unlike the original env_t2w.sh: every
# 04_train wrapper re-sources this file at its own top, so an unconditional assignment would silently
# clobber scripts/cluster/tamia_env.sh's scratch override and write checkpoints into TamIA /project
# (file-count quota) — this happened for real on the T2f port (2026-09-17). See env_t2f.sh.
#
# TRAINING_CONTRAST is "t1c" to match the contrast token used everywhere else in this dataset's eval
# configs (column_order: [t1c, t1n, t2f, t2w]) — significance's "exclude the training contrast" logic
# keys off this exact string.

export TRAINING_CONTRAST="t1c"
export nnUNet_results="${nnUNet_results:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1c/nnUNet}"

# 2500 epochs; the slowest AugLab experiments take ~90 s/epoch (srcsm ~190 s). Default ~7-day limit,
# overridable (e.g. a resume that only needs 2 days) by exporting RUN_JOB_TIME_DEFAULT beforehand.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-6-23:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
