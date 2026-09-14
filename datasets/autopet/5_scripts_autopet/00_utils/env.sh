#!/usr/bin/env bash
# Source this at the top of every autopet pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_autopet root)
#
# autopet = AutoPET whole-body FDG/PSMA PET-CT tumor-lesion segmentation. Onboarded
# 2026-09-13 as this project's first PET-modality training task, replacing an earlier
# HECKTOR plan dropped because its Grand-Challenge access-request gate wasn't worth it —
# AutoPET needs no account/DUA at all (FDAT NIfTI route, CC BY-NC 4.0).
#
# TWO COHORTS IN ONE ARCHIVE, TWO DIFFERENT ROLES (do not conflate them):
#   - FDG cohort (UKT Tübingen) = TRAINS. CT + PET are the two training modalities
#     (this file = CT default; see env_pet.sh for the PET override — same pattern as
#     chaos t1in/t2spir, brats2024-glioma t1n/t2w, open-ms flair/t1w, ispy2 t1wce/t2w).
#   - PSMA cohort (LMU Munich) = EVAL-ONLY. Different tracer (18F/68Ga-PSMA vs FDG) and
#     disease (prostate vs melanoma/lymphoma/lung), same annotation protocol. Scored with
#     the FDG-trained checkpoints, never trained on — the cross-institution/cross-tracer
#     OOD axis (05_predict's PREDICT_ITEMS_DEFAULT includes psma_ct/psma_pet as EXTRA
#     items on the SAME "own" driver, same shape as chaos's own PREDICT_ITEMS_DEFAULT
#     spanning more items than its 2 training contrasts — no separate cross-dataset shim
#     layer needed since PSMA lives in this same downloaded archive, not a separate repo).
#
# CLAIMED-N vs USABLE-N (pre-flight checklist item #2, verified from the FDAT record's own
# description, not the landing-page headline count):
#   FDG:  1014 studies / 900 patients total, but only 501 patients have an actual lesion —
#         the other 513 are NEGATIVE CONTROLS (empty mask). Only the 501 positive patients
#         go into 4_splits_autopet/ (see 01_create_splits/01_01_create_splits.py);
#         negative controls are an optional false-positive-rate side-check, never blended
#         into the headline Dice/HD95 pool.
#   PSMA: 597 studies / 378 patients, 537 positive / 60 negative controls. Same rule.
#   Some patients have multiple studies (timepoints) — 01_create_splits splits by PATIENT,
#   not study, to avoid leaking a patient's second timepoint across train/val/test.
#
# LABEL: binary tumor-lesion mask ("any tracer-avid malignant lesion"), annotated by
# radiologists (identify from PET+CT+clinical report, then manual free-hand segmentation
# in axial slices) — appearance-defined (PET uptake / CT enhancement pattern), NOT
# boundary-defined. Groups with open-ms/BraTS on the causal-ablation ladder, not CHAOS.
#
# LICENSE: CC BY-NC 4.0 (FDAT NIfTI route). Fine for this project's research/paper use
# (same precedent as duke-breast-mri) but NOT eligible for the CC-BY-only git-annex public
# upload list if that's ever revisited.
#
# Source archive: FDAT record rdkqd-wdh87 (DOI 10.57754/FDAT.rdkqd-wdh87), title
# "PSMA-FDG-PET-CT-Lesions", v2 (corrected masks). Already in nnU-Net raw layout
# (imagesTr/<tracer>_<patient>_<study>_000{0,1}.nii.gz — 0000=CT resampled to PET grid,
# 0001=PET in SUV; labelsTr/; dataset.json; splits_final.json). That splits_final.json is
# the CHALLENGE's own 5-fold CV over ALL data (positives + negative controls mixed, no
# sealed test) — NOT this project's split. 02_nnunet/02_00_convert.py splits the 2-channel
# archive into two single-channel nnU-Net Datasets (120=CT, 121=PET) using ONLY this
# project's own split.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="autopet"
export NNUNET_DATASET_ID="Dataset120_AutoPET_CT"
export MODEL_TYPE="autopet_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-ct}"

# EPOCH POLICY: 2000 (user-confirmed 2026-09-13 — matches on-harmony/open-ms, NOT
# toothfairy2's 1000 or chaos's 200).
export NNUNET_NUM_EPOCHS_DEFAULT="${NNUNET_NUM_EPOCHS_DEFAULT:-2000}"

# Training runs on TamIA (whole-node H100) per project standing default — see CLAUDE.md
# "Slurm jobs go to TamIA, not Vulcan." 60h placeholder walltime; RE-TIME from a real
# per-epoch measurement once data lands (sizing probe), do not trust this number blind —
# whole-body PET/CT volumes are far larger than any other dataset here (skull-base to
# mid-thigh vs. e.g. open-ms's brain-only FOV), so per-epoch cost is genuinely unknown
# until measured.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-60:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# No BIDS conversion needed — the FDAT archive already ships nnU-Net raw NIfTI directly
# (see 02_nnunet/02_00_convert.py's docstring for why 1_BIDS_autopet/ stays empty).

export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_autopet/02_metrics}"
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_autopet/01_predictions/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_autopet}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_autopet}"

# Raw archive staging location (outside git, huge single zip) — used by 02_00_convert.py.
export AUTOPET_ARCHIVE_ZIP="${AUTOPET_ARCHIVE_ZIP:-/scratch/paulh/pet_task_staging/autopet/psma-fdg-pet-ct-lesions_v2.zip}"
export AUTOPET_ARCHIVE_EXTRACT_DIR="${AUTOPET_ARCHIVE_EXTRACT_DIR:-/scratch/paulh/pet_task_staging/autopet/extracted}"
