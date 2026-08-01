#!/usr/bin/env bash
# Source this file at the top of every kidney-t2w pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_kidney-t2w root)
#
# KIDNEY-T2W is an EVALUATION-ONLY dataset (see datasets/kidney-t2w/README.md). We
# never train here — we run models trained on `chaos` (MR T1-DUAL in-phase /
# T2-SPIR) over this dataset's T2-weighted kidney MRI volumes to measure MRI->MRI
# (different scanner/protocol/patient-cohort) domain-randomization generalization
# on the kidney label. Hence there is NO 01_create_splits / 03_preprocess / 04_train
# stage; the CHAOS_* vars point predict/evaluate at the chaos checkpoints.
#
# This is the kidney-t2w "config": it sets the dataset-specific values, then
# sources datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="kidney-t2w"
DATASET_ROLE="test-only"

export CHAOS_DATASET_ROOT="${CHAOS_DATASET_ROOT:-${DATASET_ROOT}/../chaos}"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_PREDICTIONS_ROOT:-${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions}"
export CHAOS_NNUNET_RAW="${CHAOS_NNUNET_RAW:-${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw}"
export CHAOS_NNUNET_PREPROCESSED="${CHAOS_NNUNET_PREPROCESSED:-${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/preprocessed}"
export CHAOS_DATASET_ID="${CHAOS_DATASET_ID:-60}"
export CHAOS_DS_NAME="${CHAOS_DS_NAME:-Dataset060_CHAOS_MR_T1in}"
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"
export CHAOS_TRAINING_CONTRAST="${CHAOS_TRAINING_CONTRAST:-t1in}"
export CHAOS_MODEL_TYPE="chaos_model"
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"

BIDS_SUBDIR="kidney-t2w"
CE_SUBDIRS="raw"
CE_EXTRA_PYTHONPATH="${CHAOS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
