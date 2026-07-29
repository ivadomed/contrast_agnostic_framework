#!/usr/bin/env bash
# Source this file at the top of every mslesseg pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_mslesseg root)
#
# MSLesSeg is an EVALUATION-ONLY dataset (see datasets/mslesseg/README.md). We never
# train here — we run models trained on `open-ms` (brain MS lesion, FLAIR-trained,
# Dataset070_OpenMS_FLAIR) over ALL 115 MSLesSeg scans (train+test archive split
# collapsed into one pool — see README "Why no split") to measure MS-lesion
# domain-randomization generalization across an independent cohort/scanner set.
# Hence there is NO 01_create_splits / 04_train stage; the OPENMS_* vars point
# predict/evaluate at the open-ms checkpoints.
#
# This is the mslesseg "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="mslesseg"
# DATASET_ROLE: test-only — no mslesseg-native models. Evaluates models from open-ms.
DATASET_ROLE="test-only"

# ── Cross-dataset model source: open-ms ──────────────────────────────────────
# open-ms trains a SINGLE model (FLAIR only) and tests cross-contrast on FLAIR/T2W/T1W
# — unlike chaos (two training contrasts, t1in/t2spir), there is only one training
# contrast here, so (unlike trusted/amos/sliver07) no env_<contrast>.sh override is
# needed. nnUNetv2_predict resolves the model from nnUNet_results + dataset id (the
# trained model dir holds plans.json/dataset.json).
#
# ${VAR:-default} guards (not plain export): predict/evaluate wrapper chains re-source
# this file PARTWAY through a run, after a cluster override (e.g.
# scripts/cluster/tamia_env_mslesseg_openmscross.sh) has already exported a
# scratch-resident path — an unconditional export would silently clobber it back to
# the git-repo-relative default. See common_env.sh for the same fix on the shared side.
export OPENMS_DATASET_ROOT="${OPENMS_DATASET_ROOT:-${DATASET_ROOT}/../open-ms}"
export OPENMS_PREDICTIONS_ROOT="${OPENMS_PREDICTIONS_ROOT:-${OPENMS_DATASET_ROOT}/8_results_open-ms/01_predictions}"
export OPENMS_NNUNET_RAW="${OPENMS_NNUNET_RAW:-${OPENMS_DATASET_ROOT}/2_nnUNet_open-ms/raw}"
export OPENMS_NNUNET_PREPROCESSED="${OPENMS_NNUNET_PREPROCESSED:-${OPENMS_DATASET_ROOT}/2_nnUNet_open-ms/preprocessed}"
export OPENMS_DATASET_ID="${OPENMS_DATASET_ID:-70}"          # numeric -- consumed as nnUNetv2_predict's -d flag
export OPENMS_DS_NAME="${OPENMS_DS_NAME:-Dataset070_OpenMS_FLAIR}"
export OPENMS_DATASET_JSON="${OPENMS_DATASET_JSON:-${OPENMS_NNUNET_RAW}/${OPENMS_DS_NAME}/dataset.json}"
export OPENMS_TRAINING_CONTRAST="${OPENMS_TRAINING_CONTRAST:-flair}"
export OPENMS_MODEL_TYPE="${OPENMS_MODEL_TYPE:-open_ms_model}"
# open-ms scripts dir on PYTHONPATH so its trainer classes (nnUNetTrainerOpenMS*)
# resolve for -tr at predict time.
OPENMS_SCRIPTS_DIR="${OPENMS_DATASET_ROOT}/5_scripts_open-ms"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="mslesseg-brain"             # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw"                          # only 0_raw exists (no preprocessed/splits — eval-only)
CE_EXTRA_PYTHONPATH="${OPENMS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
