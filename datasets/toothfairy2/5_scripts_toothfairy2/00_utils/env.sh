#!/usr/bin/env bash
# Source this file at the top of every toothfairy2 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_toothfairy2 root)
#
# ToothFairy2 — maxillofacial multi-structure segmentation on dental/maxillofacial
# CBCT. Onboarded 2026-09-07 as this project's FIRST NON-CT/NON-MRI training task:
# every other training set here is MRI (on-harmony, open-ms, brats2024-glioma,
# chaos, ispy2) with CT only ever appearing as a cross-dataset TEST modality
# (amos, sliver07, msd-spleen). CBCT is a genuinely new acquisition physics —
# cone-beam geometry, heavy scatter, no calibrated Hounsfield scale, strong
# beam-hardening streaks around metal — while staying volumetric and
# anatomically framed like CT/MRI (which is why ultrasound was ruled out).
#
# WHY THIS TASK: the targets are BOUNDARY-DEFINED, not texture-defined — cortical
# bone against air/soft tissue, enamel against pulp, sinus air against bone. That
# is deliberately the opposite pole from open-ms lesions / brats tumour
# sub-regions, and it is the arm the paper's causal-ablation dissociation table
# needs more of (see CLAUDE.md, "The causal-ablation ladder": the rung-4→5
# noise-fill→real-fill step is large on texture-defined targets and near-zero on
# boundary-defined ones — chaos organs is currently the ONLY boundary-defined
# ladder, so a second, independent one is worth a lot).
#
# SINGLE TRAINING MODALITY. Unlike chaos (t1in/t2spir), brats2024-glioma
# (t1n/t2w), on-harmony (T1w/T2w), open-ms (flair/t1w) and ispy2 (t1wce/t2w),
# ToothFairy2 ships exactly one acquisition type, so there is no second training
# contrast and no env_<contrast>.sh sibling. Consequences, all deliberate:
#   - `combined_contrasts/` collapses to this single modality's own table; the
#     cross-dataset meta-heatmap takes that table directly (see
#     06_evaluate/configs/toothfairy2_combined_01_results.yaml).
#   - the cross-modality generalization axis is carried entirely by EXTERNAL test
#     sets rather than by a second in-house contrast — hanseg (CT + MR-T1 of the
#     same 42 head-and-neck patients) is the companion, sharing the `mandible`
#     class only. Same shape as chaos → amos/sliver07 (liver only).
#
# LICENSE: CC BY-SA 4.0 (ToothFairy2 challenge, Univ. of Modena and Reggio Emilia
# + Radboud UMC). Redistributable with attribution + share-alike — so unlike
# LLD-MMRI (the DUA that helped sink the liver-HCC extension, see CLAUDE.md's
# atlas-liver-hcc exclusion), this one is a legitimate future git-annex upload
# candidate and qualitative figures showing its images are unambiguously fine.
#
# This is the toothfairy2 "config": it sets the dataset-specific values, then
# sources datasets/00_commun_scripts/00_00_utils/common_env.sh, which derives
# every standard 9-subdir path from $DATASET_NAME.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="toothfairy2"
export NNUNET_DATASET_ID="Dataset110_ToothFairy2CBCT"
export MODEL_TYPE="toothfairy2_model"
# Only one training modality exists (see header). Kept as a variable rather than
# inlined so the standard results layout
# (01_predictions/<model>/<contrast>/<category>/<run>) is spelled the same way as
# every other dataset and the shared drivers need no special-casing.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-cbct}"

# Single source of truth for the nnU-Net dataset id — every 04/05/06 script reads
# this instead of hardcoding it (the improvement ispy2 made over ambl, which
# repeated "090"/"091" literally in ~40 files).
export DATASET_ID_CBCT="${DATASET_ID_CBCT:-110}"

# common_env config (plain vars — consumed by common_env, not exported):
BIDS_SUBDIR="maxillofacial-toothfairy2"   # → BIDS_ROOT under 1_BIDS_toothfairy2/
CE_SUBDIRS="raw preprocessed splits"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# EPOCH POLICY: 1000. Not a copy of any existing dataset's number — chaos's 200 is
# tuned to its 20-case cohort, brats's 2500 / on-harmony's + open-ms's 2000 to
# theirs. ToothFairy2 contributes ~407 training-pool cases (see
# 01_create_splits/01_01_create_splits.py), the largest training cohort in the
# project after ispy2, and 1000 matches the horizon ispy2 settled on for a cohort
# of that size. Re-time from a real TamIA per-epoch measurement before trusting
# RUN_JOB_TIME_DEFAULT; do not assume another dataset's per-epoch cost transfers
# (different patch size from nnU-Net's own plan, different voxel count).
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-48:00:00}"

# nnUNet_results: checkpoint home for nnUNet-category methods. AugLab wrappers
# override NNUNET_RESULTS_BASE → .../<contrast>/auglab. Guarded (never a plain
# `export VAR=default`) so a cluster override file (scripts/cluster/
# tamia_env_toothfairy2.sh), sourced AFTER this file, survives the re-source that
# 04_00_common.sh performs — the bug that sent two ambl runs' checkpoints to the
# wrong filesystem.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct/nnUNet}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_toothfairy2}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_toothfairy2}"
