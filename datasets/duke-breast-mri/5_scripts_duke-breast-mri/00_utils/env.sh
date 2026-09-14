#!/usr/bin/env bash
# Source this at the top of every duke-breast-mri pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_duke-breast-mri root)
#
# duke-breast-mri = Duke-Breast-Cancer-MRI (TCIA), post-contrast T1w DCE breast-
# tumour segmentation, EVAL-ONLY cross-dataset test set for I-SPY2-trained models
# (added 2026-09-06 as a second, larger OOD counterweight to ambl -- see the
# coordinator brief / project decision log). Single modality only (t1wce-analog);
# Duke has no T2w acquisition in this cohort, so there is no t2w branch here (unlike
# ambl/chaos/brats2024-glioma/on-harmony/open-ms's two-training-modality shape).
#
# Masks: MAMA-MIA (Garrucho et al. 2025, Synapse syn60868042) expert tumour
# segmentations for the Duke-only subset (291 patients) -- NOT TCIA's own
# Annotation_Boxes.csv (those are 3D bounding boxes only, no per-voxel mask).
# License: CC BY-NC 4.0 (Duke-Breast-Cancer-MRI's own TCIA license, inherited by
# MAMA-MIA). Fine for this project's research use (same precedent as ATLAS-Liver-
# HCC's CC BY-NC-SA 4.0 being ruled "fine" -- see CLAUDE.md's Atlas-Liver-HCC
# exclusion section) but NOT eligible for the project's public CC-BY-only
# git-annex redistribution list -- flag permanently, do not silently include it
# there later.
#
# Like ambl (post-ROLE_CHANGE_20260904), this dataset never trains -- it is a
# pure external test set. No DatasetXXX nnU-Net id is allocated; test images/
# labels live flat under 2_nnUNet_duke-breast-mri/raw/{imagesTs,labelsTs}_t1wce/
# (same convention ambl/cirrmri-liver/kidney-t2w use for eval-only sets).

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="duke-breast-mri"
export MODEL_TYPE="duke_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"

# CPU-only dataset (no training/preprocessing GPU jobs originate here), but keep
# a sane default in case a future step needs it.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-04:00:00}"

# common_env config (plain vars -- consumed by common_env, not exported to the env):
export BIDS_SUBDIR="breast-duke"
CE_SUBDIRS="preprocessed splits"          # 2_nnUNet/preprocessed + 4_splits (no 0_raw consumer)
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_duke-breast-mri/02_metrics}"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_duke-breast-mri"
export RESULTS_DIR="${DATASET_ROOT}/8_results_duke-breast-mri"

# ── Cross-dataset model source: ispy2 (eval-only, same shape as ambl's block) ──
export ISPY2_DATASET_ROOT="${ISPY2_DATASET_ROOT:-${DATASET_ROOT}/../ispy2}"
export ISPY2_PREDICTIONS_ROOT="${ISPY2_PREDICTIONS_ROOT:-${ISPY2_DATASET_ROOT}/8_results_ispy2/01_predictions}"
export ISPY2_NNUNET_RAW="${ISPY2_NNUNET_RAW:-${ISPY2_DATASET_ROOT}/2_nnUNet_ispy2/raw}"
export ISPY2_NNUNET_PREPROCESSED="${ISPY2_NNUNET_PREPROCESSED:-${ISPY2_DATASET_ROOT}/2_nnUNet_ispy2/preprocessed}"
export ISPY2_DATASET_ID="${ISPY2_DATASET_ID:-100}"
export ISPY2_TRAINING_CONTRAST="${ISPY2_TRAINING_CONTRAST:-t1wce}"
export ISPY2_MODEL_TYPE="ispy2_model"
# Identical label numbering to ispy2 (background=0, tumour=1) -- no --label_map needed.
export ISPY2_DATASET_JSON="${ISPY2_NNUNET_RAW}/Dataset100_ISPY2T1wce/dataset.json"
ISPY2_SCRIPTS_DIR="${ISPY2_DATASET_ROOT}/5_scripts_ispy2"
export PYTHONPATH="${ISPY2_SCRIPTS_DIR}:${PYTHONPATH:-}"
