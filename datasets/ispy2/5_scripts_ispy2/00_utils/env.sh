#!/usr/bin/env bash
# Source this file at the top of every ispy2 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_ispy2 root)
#
# I-SPY2 is a cross-evaluation-only dataset for this project's breast-tumor
# segmentation task (see datasets/ispy2/1_BIDS_ispy2/breast-ispy2/README):
# it is meant to be evaluated with a model trained on `ambl`
# (Advanced-MRI-Breast-Lesions), the same way MSLesSeg/ms3seg evaluate
# open-ms-trained models. We never train here.
#
# WIRED 2026-09-04: cross-dataset predict/evaluate against ambl now landed (see
# 05_predict/05_01_predict_common.sh, SOURCE_PREFIX="AMBL"). Same pattern
# mslesseg -> open-ms uses (OPENMS_*-style vars). ONLY the 122 usable BILATERAL
# cases (of 560 with BIDS data) are ever fed to prediction — see
# 4_splits_ispy2/unilateral_fov_exclusions.json and 05_predict/05_00_build_test_
# inputs.py, which is the single place this filter is applied; nothing downstream
# re-derives the case list.
#
# nnUNet_raw/preprocessed here (unqualified, common_env.sh defaults) hold ONLY
# the flat imagesTs_<item>/labelsTs_<item> test-input dirs 05_00 builds — there is
# no Dataset<id>/ subdir and no training data, matching every other cross-
# dataset-only consumer (lld-mmri-hcc/liverhccseg -> atlas-liver-hcc, mslesseg ->
# open-ms). The AMBL_* vars below point at the actual TRAINED model — its own
# nnUNet_raw/preprocessed/checkpoints — which on TamIA live only under
# /scratch/p/paulh/ambl/... (see scripts/cluster/tamia_env_ispy2.sh, sourced
# AFTER this file to override every AMBL_* path to scratch).
#
# This is the ispy2 "config": it sets the dataset-specific values, then
# sources datasets/00_commun_scripts/00_00_utils/common_env.sh for the
# standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="ispy2"
# DATASET_ROLE (CHANGED 2026-09-04): was "test-only"; I-SPY2 is now this project's
# PRIMARY breast-cancer TRAINING dataset (it replaces ambl, whose 99-case cohort
# overfit), while ambl becomes the external test-only set. Both roles now coexist
# in this file: the training block at the bottom is new, and the AMBL_* cross-
# dataset block above is kept intact so the existing 05_0X_predict_ambl_*.sh
# wrappers (ambl models -> ispy2 data) keep working unchanged.
DATASET_ROLE="both"

# ── Cross-dataset model source: ambl ────────────────────────────────────────
# ambl trains TWO modalities (t1wce=Dataset090, t2w=Dataset091, unlike atlas-
# liver-hcc's single modality) -- AMBL_TRAINING_CONTRAST/AMBL_DATASET_ID are
# left unset here (default git-repo-relative fallback below assumes t1wce;
# each t2w-trained method wrapper overrides both explicitly before sourcing
# 05_01_predict_common.sh, mirroring ambl's own env_t2w.sh pattern).
export AMBL_DATASET_ROOT="${AMBL_DATASET_ROOT:-${DATASET_ROOT}/../ambl}"
export AMBL_PREDICTIONS_ROOT="${AMBL_PREDICTIONS_ROOT:-${AMBL_DATASET_ROOT}/8_results_ambl/01_predictions}"
export AMBL_NNUNET_RAW="${AMBL_NNUNET_RAW:-${AMBL_DATASET_ROOT}/2_nnUNet_ambl/raw}"
export AMBL_NNUNET_PREPROCESSED="${AMBL_NNUNET_PREPROCESSED:-${AMBL_DATASET_ROOT}/2_nnUNet_ambl/preprocessed}"
export AMBL_DATASET_ID="${AMBL_DATASET_ID:-90}"
export AMBL_TRAINING_CONTRAST="${AMBL_TRAINING_CONTRAST:-t1wce}"
export AMBL_MODEL_TYPE="ambl_model"
# ambl's dataset.json (label map: background 0, tumour 1) -- ispy2's own GT masks
# already use the same 0/1 numbering (lesion=1), so evaluate needs no --label_map
# remap (unlike lld-mmri-hcc -> atlas-liver-hcc, which needed one).
export AMBL_DATASET_JSON="${AMBL_NNUNET_RAW}/Dataset090_AMBLT1wce/dataset.json"
# ambl scripts dir on PYTHONPATH so its trainer classes (nnUNetTrainerAMBL*)
# resolve for -tr at predict time if not already registered in the venv.
AMBL_SCRIPTS_DIR="${AMBL_DATASET_ROOT}/5_scripts_ambl"

# common_env config (plain vars -- consumed by common_env, not exported to the env):
BIDS_SUBDIR="breast-ispy2"               # -> BIDS_ROOT under 1_BIDS_ispy2/
# CHANGED with DATASET_ROLE above: ispy2 now trains here, so it needs
# nnUNet_preprocessed (2_nnUNet_ispy2/preprocessed) and SPLITS_DIR
# (4_splits_ispy2) exported as well. "raw" is deliberately absent: the pipeline
# reads exclusively from 1_BIDS_ispy2/, never 0_raw_ispy2/ (same as ambl).
CE_SUBDIRS="preprocessed splits"
CE_EXTRA_PYTHONPATH="${AMBL_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING ROLE (added 2026-09-04) — ispy2 as the PRIMARY breast task
# ═══════════════════════════════════════════════════════════════════════════
# Mirrors datasets/ambl/5_scripts_ambl/00_utils/env.sh, which is the closest
# working template: a two-training-contrast (t1wce + t2w) tumour-segmentation
# MRI dataset with the full 6-method suite + causal-ablation ladder. As in ambl
# (and chaos t1in/t2spir, brats2024-glioma t1n/t2w), TRAINING_CONTRAST switches
# between the two via env_t2w.sh, each modality gets its OWN nnU-Net Dataset id,
# and results are pooled later via 06_12_combined_modality_summary.sh.
#
# STANDARD results layout (chaos / on-harmony / open-ms / brats2024-glioma / ambl):
#   01_predictions/<MODEL_TYPE>/<TRAINING_CONTRAST>/<nnUNet|auglab>/<RUN_ID>/
#   02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/
# nnUNet_results points at the nnUNet-category base; auglab train wrappers
# override NNUNET_RESULTS_BASE -> .../auglab.

# PROJECT_TODO (data build): the nnU-Net Dataset IDs for ispy2 are NOT yet
# decided — they are being chosen by the parallel data-build effort that owns
# 02_nnunet/. 100/101 below are PLACEHOLDERS chosen to avoid every id currently
# in use across the repo (030-032, 051-052, 060-061, 070-071, 080, 090-091).
# Confirm against 02_nnunet/'s real converters and fix HERE ONLY — every 04/05/06
# script reads these two variables instead of hardcoding an id (a deliberate
# improvement over ambl, which repeats "090"/"091" in ~40 files).
export DATASET_ID_T1WCE="${DATASET_ID_T1WCE:-100}"
export DATASET_ID_T2W="${DATASET_ID_T2W:-101}"
# Generic/base id (matches brats2024-glioma's env.sh convention) — real training
# always passes DATASET_ID explicitly from the two vars above.
# PROJECT_TODO: confirm the Dataset<id>_<Name> suffix the converters actually write.
export NNUNET_DATASET_ID="Dataset${DATASET_ID_T1WCE}_ISPY2"
export MODEL_TYPE="ispy2_model"

# Training contrast: t1wce (early post-contrast T1 DCE, default) or t2w.
# Conditional so env_t2w.sh can pre-export it before this file (re-)sources.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"

# EPOCH POLICY: 1000 (see 04_train/04_00_common.sh) — a deliberate, user-set
# decision for this retrain, NOT ambl's 2000. 48h gives headroom at that horizon
# on a shared GPU; on TamIA the whole-node pack chain caps each job at 23:59:00
# and resumes from checkpoint_latest, so this value only bites the Vulcan/
# Killarney one-job-per-fold path.
# PROJECT_TODO: re-time once a real per-epoch cost is measured on ispy2 (its
# cohort is much larger than ambl's 99 patients — don't assume ambl's numbers).
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-48:00:00}"

# nnUNet_results: checkpoint home for nnUNet-category methods. Path includes
# model_type + training_contrast so each training context is isolated. AugLab
# wrappers override NNUNET_RESULTS_BASE to .../<contrast>/auglab.
# Conditional so env_t2w.sh's pre-export (set before this file is re-sourced by
# 04_00_common.sh) wins — and so a cluster override file (tamia_env_ispy2.sh)
# is not clobbered on a re-source. NOTE: ambl's env_t2w.sh exported this
# UNCONDITIONALLY and that silently undid tamia_env_ambl.sh's scratch override
# inside training subshells, sending 2 runs' checkpoints to the wrong filesystem
# (see ambl 05_predict/05_20's header, gotcha #2). Both files here are guarded.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_ispy2/01_predictions/ispy2_model/t1wce/nnUNet}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_ispy2}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_ispy2}"
