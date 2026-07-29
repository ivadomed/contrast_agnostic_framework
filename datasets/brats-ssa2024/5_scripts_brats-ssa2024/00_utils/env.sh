#!/usr/bin/env bash
# Source this file at the top of every brats-ssa2024 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_brats-ssa2024 root)
#
# BraTS-SSA 2024 (Sub-Saharan Africa glioma cohort, kagglehub kaalmurlidhar/brats2024-africa,
# "95_Glioma" subset only -- see README.md) is an EVALUATION-ONLY dataset. We never train
# here -- we run models trained on `brats2024-glioma` (both its T1n- and T2w-trained
# families) over ALL 95 glioma cases to measure glioma-segmentation domain-randomization
# generalization to an independent cohort/scanner/population (Sub-Saharan Africa vs the
# original BraTS adult glioma cohort). Hence there is NO 01_create_splits / 04_train stage;
# the BRATS_* vars point predict/evaluate at the brats2024-glioma checkpoints.
#
# Label note: BraTS-SSA is PRE-TREATMENT (no resection cavity) -- GT has only 3 of
# brats2024-glioma's 4 foreground labels (NCR=1, SNFH=2, ET=3; RC=4 never present).
# Verified empirically on the raw seg.nii files (see 00_utils/00_00_ingest_and_bidsify.py).
# 06_evaluate restricts scoring to these 3 labels explicitly (--labels NCR SNFH ET).
#
# This is the brats-ssa2024 "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="brats-ssa2024"
# DATASET_ROLE: test-only -- no brats-ssa2024-native models. Evaluates models from brats2024-glioma.
DATASET_ROLE="test-only"

# -- Cross-dataset model source: brats2024-glioma --------------------------------
# brats2024-glioma trains TWO separate model families (T1n and T2w) -- like chaos's
# t1in/t2spir -- so BRATS_TRAINING_CONTRAST/_DATASET_ID/_DS_NAME/_DATASET_JSON are
# switchable per-contrast (default t1n; env_t2w.sh pre-exports the T2w variant).
# ${VAR:-default} guards throughout: predict/evaluate wrapper chains re-source this
# file partway through a run (see common_env.sh's fix, 2026-07-29) -- an unconditional
# export would silently clobber an already-exported cluster/contrast override.
export BRATS_DATASET_ROOT="${BRATS_DATASET_ROOT:-${DATASET_ROOT}/../brats2024-glioma}"
export BRATS_PREDICTIONS_ROOT="${BRATS_PREDICTIONS_ROOT:-${BRATS_DATASET_ROOT}/8_results_brats2024-glioma/01_predictions}"
export BRATS_NNUNET_RAW="${BRATS_NNUNET_RAW:-${BRATS_DATASET_ROOT}/2_nnUNet_brats2024-glioma/raw}"
export BRATS_NNUNET_PREPROCESSED="${BRATS_NNUNET_PREPROCESSED:-${BRATS_DATASET_ROOT}/2_nnUNet_brats2024-glioma/preprocessed}"
export BRATS_DATASET_ID="${BRATS_DATASET_ID:-051}"
export BRATS_DS_NAME="${BRATS_DS_NAME:-Dataset051_BraTS2024GliomaT1n}"
export BRATS_DATASET_JSON="${BRATS_DATASET_JSON:-${BRATS_NNUNET_RAW}/${BRATS_DS_NAME}/dataset.json}"
export BRATS_TRAINING_CONTRAST="${BRATS_TRAINING_CONTRAST:-t1n}"
export BRATS_MODEL_TYPE="${BRATS_MODEL_TYPE:-brats2024_glioma_model}"
# brats2024-glioma scripts dir on PYTHONPATH so its trainer classes
# (nnUNetTrainerBraTS2024Glioma*) resolve for -tr at predict time.
BRATS_SCRIPTS_DIR="${BRATS_DATASET_ROOT}/5_scripts_brats2024-glioma"

# common_env config (plain vars -- consumed by common_env, not exported to the env):
BIDS_SUBDIR="ssa-brain-brats2024"        # -> BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw"                          # only 0_raw exists (no preprocessed/splits -- eval-only)
CE_EXTRA_PYTHONPATH="${BRATS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
