#!/usr/bin/env bash
# Source at the top of every isles2022 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#
# isles2022 = ISLES'22 training release (Zenodo 10.5281/zenodo.7153326): 250 acute/subacute
# ischemic-stroke MRIs, 246 usable (3 empty masks + 1 case whose FLAIR FOV cuts the lesion). Two TRAINING contrasts, each its own
# nnU-Net Dataset, each tested CROSS-CONTRAST on held-out patients' DWI and FLAIR (each model is trained on one, tested on both):
#   dwi   (Dataset140_ISLES2022_DWI)    -- b=1000 DWI (the contrast the labels were drawn on)
#   flair (Dataset141_ISLES2022_FLAIR)  -- FLAIR resampled onto the DWI grid
# <<ref-only
# ADC (also in the ISLES release, derived from DWI) is deliberately NOT used anywhere in this benchmark (Paul, 2026-10-04): DWI is the
# contrast the labels were drawn on and ADC is nearly redundant with it. It stays in the BIDS tree only because that is a faithful copy.
# ref-only>>
# Standard results layout (same as ispy2/chaos/brats):
#   8_results_isles2022/01_predictions/isles2022_model/<contrast>/<nnUNet|auglab>/<RUN_ID>/
#   8_results_isles2022/02_metrics/isles2022_model/<contrast>/
# License: CC BY 4.0 + source terms (no redistribution without ISLES'22 team's written OK).

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="isles2022"
export DATASET_ROLE="training"
export MODEL_TYPE="isles2022_model"
export BIDS_SUBDIR="stroke-brain-isles2022"
CE_SUBDIRS="preprocessed splits"

export DATASET_ID_DWI="${DATASET_ID_DWI:-140}"
export DATASET_ID_FLAIR="${DATASET_ID_FLAIR:-141}"
export NNUNET_DATASET_ID="Dataset${DATASET_ID_DWI}_ISLES2022_DWI"
# Training contrast: dwi (default) or flair; env_flair.sh pre-exports flair.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-dwi}"

# Small volumes (~112x112x73 @2mm): don't take the L40S 16 CPU/110G-per-GPU defaults.
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}"
export RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-48G}"
# PROJECT_TODO: re-time from a real per-epoch measurement (TamIA sizing probe) before trusting.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-48:00:00}"

source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"

# nnUNet_results is DERIVED here from PREDICTIONS_ROOT (guarded in common_env.sh, so a cluster override file wins) and TRAINING_CONTRAST,
# every time this file is sourced. Never export it unconditionally elsewhere (env_flair.sh used to: a wrapper re-sourcing it silently sent
# nnUNet-category checkpoints to the repo path instead of the TamIA scratch override -- the ambl bug) and never leave a stale value from
# another contrast: it must always agree with PREDICTIONS_ROOT + MODEL_TYPE + TRAINING_CONTRAST.
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_isles2022}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_isles2022}"
