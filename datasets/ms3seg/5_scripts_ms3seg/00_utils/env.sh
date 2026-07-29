#!/usr/bin/env bash
# Source this file at the top of every ms3seg pipeline script.
#
# MS3SEG (Bawil et al., figshare 10.6084/m9.figshare.30393475.v6, CC-BY-4.0) is an
# EVALUATION-ONLY dataset. We never train here -- we run models trained on `open-ms`
# (brain MS lesion, both FLAIR- and T1w-trained families) over all 100 patients to
# measure MS-lesion domain-randomization generalization to an independent cohort AND
# scanner vendor (Toshiba -- open-ms/mslesseg are not Toshiba). Hence no
# 01_create_splits / 04_train stage; the OPENMS_* vars point predict/evaluate at the
# open-ms checkpoints.
#
# Label note: MS3SEG ships a tri-mask (ventricles=64, normal WMH=191, MS lesion=255 --
# raw uint8 values, verified empirically; NOT small class indices, and the source
# NIfTI headers have a broken scl_slope=NaN that makes nibabel's default read return
# garbage-scaled values -- 00_utils/00_00_ingest_and_bidsify.py reads unscaled and
# rewrites a clean header). open-ms's own label space is {background:0, lesion:1}, so
# evaluation uses a cross-label-space --label_map (pred_id=1 -> gt_id=255), scoring
# ONLY the MS-lesion sub-mask -- ventricles/normal-WMH are structurally different
# classes open-ms's model was never trained to predict and are excluded from scoring
# (not counted as false positives), same spirit as BraTS-SSA's --labels subset.
#
# This is the ms3seg "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="ms3seg"
DATASET_ROLE="test-only"

# -- Cross-dataset model source: open-ms (FLAIR-trained by default; env_t1w.sh switches
# to the T1w-trained family). ${VAR:-default} guards: predict/evaluate wrapper chains
# re-source this file partway through a run -- an unconditional export would silently
# clobber a cluster override (scripts/cluster/tamia_env_*.sh) sourced beforehand.
export OPENMS_DATASET_ROOT="${OPENMS_DATASET_ROOT:-${DATASET_ROOT}/../open-ms}"
export OPENMS_PREDICTIONS_ROOT="${OPENMS_PREDICTIONS_ROOT:-${OPENMS_DATASET_ROOT}/8_results_open-ms/01_predictions}"
export OPENMS_NNUNET_RAW="${OPENMS_NNUNET_RAW:-${OPENMS_DATASET_ROOT}/2_nnUNet_open-ms/raw}"
export OPENMS_NNUNET_PREPROCESSED="${OPENMS_NNUNET_PREPROCESSED:-${OPENMS_DATASET_ROOT}/2_nnUNet_open-ms/preprocessed}"
export OPENMS_DATASET_ID="${OPENMS_DATASET_ID:-70}"
export OPENMS_DS_NAME="${OPENMS_DS_NAME:-Dataset070_OpenMS_FLAIR}"
export OPENMS_DATASET_JSON="${OPENMS_DATASET_JSON:-${OPENMS_NNUNET_RAW}/${OPENMS_DS_NAME}/dataset.json}"
export OPENMS_TRAINING_CONTRAST="${OPENMS_TRAINING_CONTRAST:-flair}"
export OPENMS_MODEL_TYPE="${OPENMS_MODEL_TYPE:-open_ms_model}"
OPENMS_SCRIPTS_DIR="${OPENMS_DATASET_ROOT}/5_scripts_open-ms"

BIDS_SUBDIR="ms3seg-brain"
CE_SUBDIRS="raw"
CE_EXTRA_PYTHONPATH="${OPENMS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
