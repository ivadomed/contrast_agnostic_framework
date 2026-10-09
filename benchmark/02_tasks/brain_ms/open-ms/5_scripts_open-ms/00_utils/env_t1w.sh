#!/usr/bin/env bash
# T1w training environment for open-ms — source this INSTEAD of env.sh for T1w
# experiments. Sources env.sh first (all shared paths), then overrides the
# contrast-specific vars. Mirrors CHAOS's env_t2spir.sh (the second-contrast pattern).
#
# The model is trained on T1w with synthesis augmentation and tested cross-contrast on
# FLAIR / T2W / T1W (all co-registered to FLAIR, so one consensus mask serves every
# contrast) — the "train one contrast, generalise to others" setup.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_t1w.sh"

# Pre-export the contrast-specific vars BEFORE sourcing env.sh so they survive the
# `:-` defaults there (and the re-source that 04_00_common.sh does).
export TRAINING_CONTRAST="t1w"
# 30h walltime for T1w folds (per request). env.sh's `:-` keeps this; if a fold hits the
# wall before 2000 epochs, resume with the same RUN_ID (--c, auto from checkpoint_latest).
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-40:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# env.sh rebuilds nnUNet_results from TRAINING_CONTRAST=t1w → .../t1w/nnUNet already.
# Only NNUNET_DATASET_ID must be overridden to the T1w nnUNet dataset (env.sh hardcodes
# the FLAIR one). Note: training keys off the numeric DATASET_ID (071) set in each
# wrapper, not this var — this keeps convert/preprocess pointed at the right dataset.
export NNUNET_DATASET_ID="Dataset071_OpenMS_T1W"
