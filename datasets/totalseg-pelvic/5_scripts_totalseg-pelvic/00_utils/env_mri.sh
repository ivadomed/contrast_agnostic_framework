#!/usr/bin/env bash
# MRI training environment for totalseg-pelvic — source this INSTEAD of env.sh for MRI
# experiments. Sources env.sh first (all shared paths), then overrides the
# contrast-specific vars. Mirrors chaos's env_t2spir.sh / autopet's env_pet.sh (the
# second-contrast pattern).
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_mri.sh"

export TRAINING_CONTRAST="mri"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-24:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# env.sh rebuilds nnUNet_results from TRAINING_CONTRAST=mri -> .../mri/nnUNet already.
# Only NNUNET_DATASET_ID must be overridden to the MRI nnUNet dataset (env.sh hardcodes
# the CT one). Training keys off the numeric DATASET_ID (131) set in each wrapper, not
# this var — this keeps convert/preprocess pointed at the right dataset.
export NNUNET_DATASET_ID="Dataset131_TotalsegPelvic_MRI"
