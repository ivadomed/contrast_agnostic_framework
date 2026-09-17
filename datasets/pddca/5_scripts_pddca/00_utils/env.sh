#!/usr/bin/env bash
# Source this file at the top of every pddca pipeline script.
#
# PDDCA — "A Public Domain Database for Computational Anatomy" (Sharp, Zaffino,
# Fritscher, Raudaschl; v1.4.1, 2016). 48 head-and-neck CT cases re-contoured from
# the RTOG 0522 trial, distributed via imagenglab.com. Used for the MICCAI 2015
# Head & Neck Auto-Segmentation Challenge.
#
# ROLE: TEST-ONLY. Nothing is ever trained here. This is the SECOND cross-dataset CT
# arm for toothfairy2, alongside [hanseg]. Its purpose is specific: HaN-Seg is a
# single 42-patient cohort, so every CT conclusion for this task currently rests on
# one institution's data. PDDCA is an INDEPENDENT CT cohort (RTOG 0522, a US
# multi-institution trial, 370 sites) against HaN-Seg's Ljubljana single centre — so
# it tests whether the CT-arm findings replicate, most importantly the +AugLab rung's
# CT-specific −7 Dice cost.
#
# LICENCE (verified 2026-09-17, 2 of the 3 pre-flight checks positive):
#   * Underlying CT = TCIA collection "Head-Neck-Cetuximab" (RTOG 0522, 111 subjects)
#     = **CC BY 3.0**, stated on TCIA's own collection page. This is authoritative.
#   * PDDCA's re-contours: its protocol doc is titled "A Public Domain Database",
#     states the data are "made available to the public domain via TCIA", and contains
#     NO restriction language anywhere (grepped for licence/copyright/redistribute).
#   * ⚠️ There is NO bundled LICENSE file in any of the three archives, so the third
#     check is INCONCLUSIVE, not negative. Do NOT claim "CC BY 4.0" — a web search
#     asserts it and no primary source confirms it.
#   Far more permissive than hanseg's CC BY-NC-ND. CC BY 3.0 attribution applies to
#   the images for any redistribution.
#
# USABLE N: **40 of 48**. All 48 case dirs carry img.nrrd; 8 have no Mandible.nrrd
# (0522c0329, 0330, 0427, 0433, 0441, 0455, 0457, 0479) and are skipped. The
# challenge test cases DO carry mandible labels — the protocol doc's phrasing
# ("*Images* of datasets ... provided as test set") suggests otherwise; counting
# disproved it. Always count, never infer.
#
# SHARED CLASS: `mandible` only, and the convention matches hanseg EXACTLY.
# PDDCA's own protocol doc, verbatim figure caption: "Example of mandible
# segmentation. Only the bone is segmented, while the teeth are excluded." Verified
# empirically too: p99 = 1726 HU inside the mask (cortical bone, no enamel
# population), 99.5% of enamel-range voxels lie OUTSIDE it, and the per-slice
# hole-fill ratio is 1.0000 — a solid bone envelope with sockets NOT carved, the
# identical signature to hanseg's Bone_Mandible. So both CT sources share one label
# convention and carry the SAME ~2% root under-coverage against toothfairy2's
# `mandible` (which does carve the sockets, fill ratio 1.073). Scoring is
# MANDIBLE-ONLY, one-to-one — never the old union, which was disproven 2026-09-17.
#
# ⚠️⚠️ FIELD OF VIEW is the same central risk as hanseg, handled the same way.
# A head-and-neck RT planning scan images the whole head and neck; a dental CBCT
# images a ~51-82 mm slab around the jaws. 01_prepare/01_01_prepare_ct.py crops every
# volume to the SAME fixed-size, mandible-centred box hanseg uses (144x128x80 mm,
# near toothfairy2's training p90) so the two CT arms are FOV-matched to each other
# as well as to training. The crop is GT-CENTRED but FIXED-SIZE and applied
# identically to every method, so position leaks while extent does not and the
# between-method comparison is unbiased — absolute Dice is NOT clinical performance.
#
# ⚠️ RESOLUTION CAVEAT, genuinely different from hanseg: PDDCA is ~1.0 x 1.0 x
# 2.5-3.0 mm (in-plane 0.94-1.18, z 2.0-3.0) against hanseg CT's 0.558 x 0.558 x 2.0
# and toothfairy2's 0.6 mm isotropic training. It is ~2x coarser in-plane. Absolute
# PDDCA Dice may therefore sit below hanseg_ct for resolution reasons rather than
# method reasons. This applies identically to every method, so between-method
# comparison stays fair — but do not read a pddca-vs-hanseg_ct gap as a cohort effect.
#
# ⚠️ POOLING: pddca_ct and hanseg_ct are BOTH CT. Wiring this in without
# `contrast_groups` pooling silently reweights the OOD estimand from 1/2 CT to 2/3 CT,
# which would amplify the +AugLab CT cost and shift the ladder with no method change.
# Pool {hanseg_ct, pddca_ct} -> CT and {hanseg_mrt1} -> MR, the same mechanism chaos
# uses to stop CT outweighing MR 3-to-1.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="pddca"
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

BIDS_SUBDIR="headneck-pddca"
CE_SUBDIRS="raw splits"
CE_EXTRA_PYTHONPATH="${TF2_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# Single held-out evaluation item (CT only — PDDCA has no MR).
export PDDCA_ITEMS="${PDDCA_ITEMS:-ct}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_pddca}"
