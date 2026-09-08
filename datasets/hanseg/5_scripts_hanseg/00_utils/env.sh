#!/usr/bin/env bash
# Source this file at the top of every hanseg pipeline script.
#
# HaN-Seg (Podobnik et al., Med. Phys. 2023) — 42 head-and-neck radiotherapy
# patients, each with a planning CT *and* a T1-weighted MR, 30 organs-at-risk
# delineated. Zenodo 7442914, CC BY-NC-ND 4.0.
#
# ROLE: TEST-ONLY. Nothing is ever trained here. This is the cross-MODALITY
# companion to toothfairy2 — the CBCT-trained models are evaluated on CT and on
# MR-T1, which is the ONLY way this task gets an out-of-domain axis (toothfairy2
# has a single training modality, so unlike chaos/brats/open-ms/on-harmony/ispy2
# it has no second in-house contrast). Same shape as mslesseg -> open-ms and
# lld-mmri-hcc -> atlas-liver-hcc.
#
# WHY THIS DATASET SPECIFICALLY: it is one cohort, one annotation protocol, and
# the SAME 42 patients imaged in both modalities. That means the CT-vs-MR
# comparison is a clean modality contrast with cohort and annotator held fixed —
# not two different datasets glued together, which is the failure mode that
# helped sink the liver-HCC extension (see CLAUDE.md's atlas-liver-hcc exclusion).
#
# SHARED CLASS: `mandible` only. HaN-Seg's single `Bone_Mandible` structure
# INCLUDES the lower dentition, so it corresponds to the UNION of toothfairy2's
# `mandible` + `lower_teeth` (see toothfairy2_labels.MANDIBLE_UNION_TARGET_IDS).
# That union is exact, not approximate — it is precisely why toothfairy2's task
# was reduced to the mandibular block. One-class overlap is the same situation
# chaos -> sliver07 (liver only) already handles.
#
# ⚠️⚠️ FIELD OF VIEW IS THE CENTRAL RISK HERE, NOT THE MODALITY GAP.
# A head-and-neck RT planning scan images the whole head and neck; a dental CBCT
# images a ~51-82 mm slab around the jaws. Feeding raw HaN-Seg volumes to a
# CBCT-trained model would test it on a field of view it has NEVER seen, and the
# resulting failure would be a FOV artefact masquerading as a modality-
# generalization result. So 01_prepare/01_01_fov_match.py crops every HaN-Seg
# volume to a fixed-size, mandible-centred box drawn to match the toothfairy2
# training FOV distribution — i.e. it simulates "a dental CBCT of this patient".
# See that script for the leakage analysis (the crop is GT-CENTRED but
# FIXED-SIZE, and applied identically to every method, so it cannot bias the
# between-method comparison).

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="hanseg"
DATASET_ROLE="test-only"

# ── Cross-dataset model source: toothfairy2 ─────────────────────────────────
export TF2_DATASET_ROOT="${TF2_DATASET_ROOT:-${DATASET_ROOT}/../toothfairy2}"
export TF2_PREDICTIONS_ROOT="${TF2_PREDICTIONS_ROOT:-${TF2_DATASET_ROOT}/8_results_toothfairy2/01_predictions}"
export TF2_NNUNET_RAW="${TF2_NNUNET_RAW:-${TF2_DATASET_ROOT}/2_nnUNet_toothfairy2/raw}"
export TF2_NNUNET_PREPROCESSED="${TF2_NNUNET_PREPROCESSED:-${TF2_DATASET_ROOT}/2_nnUNet_toothfairy2/preprocessed}"
export TF2_DATASET_ID="${TF2_DATASET_ID:-110}"
export TF2_TRAINING_CONTRAST="${TF2_TRAINING_CONTRAST:-cbct}"
export TF2_MODEL_TYPE="toothfairy2_model"
export TF2_DATASET_JSON="${TF2_NNUNET_RAW}/Dataset110_ToothFairy2CBCT/dataset.json"
TF2_SCRIPTS_DIR="${TF2_DATASET_ROOT}/5_scripts_toothfairy2"

BIDS_SUBDIR="headneck-hanseg"
CE_SUBDIRS="raw splits"
CE_EXTRA_PYTHONPATH="${TF2_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# The two held-out evaluation items (one per modality of the SAME patients).
export HANSEG_ITEMS="${HANSEG_ITEMS:-ct mrt1}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_hanseg}"
