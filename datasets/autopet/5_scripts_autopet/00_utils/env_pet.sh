#!/usr/bin/env bash
# PET training environment for autopet — source this INSTEAD of env.sh for PET
# experiments. Sources env.sh first (all shared paths), then overrides the
# contrast-specific vars. Mirrors open-ms's env_t1w.sh / chaos's env_t2spir.sh (the
# second-contrast pattern).
#
# The model is trained on PET (FDG, SUV-normalized) with synthesis augmentation and
# tested cross-contrast on CT (same FDG held-out patients), then cross-institution on
# the PSMA cohort's own CT + PET (05_predict) — the "train one contrast, generalise to
# others (and to a different institution/tracer)" setup.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_pet.sh"

# Pre-export the contrast-specific vars BEFORE sourcing env.sh so they survive the `:-`
# defaults there (and the re-source that 04_00_common.sh does).
export TRAINING_CONTRAST="pet"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-60:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# env.sh rebuilds nnUNet_results from TRAINING_CONTRAST=pet -> .../pet/nnUNet already.
# Only NNUNET_DATASET_ID must be overridden to the PET nnUNet dataset (env.sh hardcodes
# the CT one). Note: training keys off the numeric DATASET_ID (121) set in each wrapper,
# not this var — this keeps convert/preprocess pointed at the right dataset.
export NNUNET_DATASET_ID="Dataset121_AutoPET_PET"
