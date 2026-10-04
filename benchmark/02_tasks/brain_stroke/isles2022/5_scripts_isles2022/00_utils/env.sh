#!/usr/bin/env bash
# Source at the top of every isles2022 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#
# isles2022 = ISLES'22 training release (Zenodo 10.5281/zenodo.7153326): 250 acute/subacute
# ischemic-stroke MRIs, 247 usable (3 empty masks). Two TRAINING contrasts, each its own
# nnU-Net Dataset, each tested CROSS-CONTRAST on held-out patients' DWI / ADC / FLAIR:
#   dwi   (Dataset140_ISLES2022_DWI)    -- b=1000 DWI (the contrast the labels were drawn on)
#   flair (Dataset141_ISLES2022_FLAIR)  -- FLAIR resampled onto the DWI grid
# ADC is derived from DWI (not an independent contrast) -> test-only, never a training contrast.
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

# Guarded (not unconditional) so env_flair.sh and cluster override files are not clobbered on re-source.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_isles2022/01_predictions/isles2022_model/dwi/nnUNet}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_isles2022}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_isles2022}"
