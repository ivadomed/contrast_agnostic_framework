#!/usr/bin/env bash
# Source at the top of every pansegdata pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#
# pansegdata = MRI part of PanSegData (OSF kysnj; Zhang et al., Med Image Anal 99:103382): pancreas MRI from 5 centers, venous-phase T1W + T2W,
# manual pancreas masks (cohort referred for pancreatic cystic lesions / suspected PDAC). 405 subjects in BIDS; 362 have BOTH contrasts; the
# 150 of them from center MCF are excluded (unreliable orientation / mask registration), so the usable set is 212 subjects (NYU 161, NWU 19,
# AHN 17, MCA 15: pool 170 + 42 held-out test); the single-contrast subjects are excluded so every contrast dataset has the same case ids.
# Two TRAINING contrasts, each its own
# nnU-Net Dataset, each tested CROSS-CONTRAST on held-out patients' T1WCE and T2W (each model trained on one, tested on both):
#   t1wce (Dataset150_PanSegData_T1WCE) -- venous-phase contrast-enhanced T1 (mask drawn on this scan)
#   t2w   (Dataset151_PanSegData_T2W)   -- T2 (own mask drawn on this scan)
# A subject's T1 and T2 masks were drawn independently: each test item is scored against ITS OWN mask.
# Standard results layout:
#   8_results_pansegdata/01_predictions/pansegdata_model/<contrast>/<nnUNet|auglab>/<RUN_ID>/
#   8_results_pansegdata/02_metrics/pansegdata_model/<contrast>/
# License: CC BY-NC 4.0 (non-commercial; redistribution with attribution allowed).

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="pansegdata"
export DATASET_ROLE="training"
export MODEL_TYPE="pansegdata_model"
export BIDS_SUBDIR="onc-pancreas-pansegdata"
CE_SUBDIRS="preprocessed splits"

export DATASET_ID_T1WCE="${DATASET_ID_T1WCE:-150}"
export DATASET_ID_T2W="${DATASET_ID_T2W:-151}"
export NNUNET_DATASET_ID="Dataset${DATASET_ID_T1WCE}_PanSegData_T1WCE"
# Training contrast: t1wce (default) or t2w; env_t2w.sh pre-exports t2w.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"

# PROJECT_TODO: re-size from a real probe on the target cluster (Killarney) before trusting these.
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}"
export RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-48G}"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-48:00:00}"

source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"

# nnUNet_results is DERIVED here from PREDICTIONS_ROOT (guarded in common_env.sh, so a cluster override file wins) and TRAINING_CONTRAST,
# every time this file is sourced; it must always agree with PREDICTIONS_ROOT + MODEL_TYPE + TRAINING_CONTRAST (never export it elsewhere).
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_pansegdata}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_pansegdata}"
