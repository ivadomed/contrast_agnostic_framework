#!/usr/bin/env bash
# Source this file at the top of every sliver07 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_sliver07 root)
#
# SLIVER07 is an EVALUATION-ONLY dataset (see datasets/sliver07/README.md). We never
# train here — we run models trained on `chaos` (MR T1-DUAL in-phase) over SLIVER07's
# 20 labeled CT volumes to measure MR→CT domain-randomization generalization. Hence
# there is NO 01_create_splits / 03_preprocess / 04_train stage (so no preprocessed/
# splits subdirs); the CHAOS_* vars point predict/evaluate at the chaos checkpoints.
#
# This is the sliver07 "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="sliver07"
# DATASET_ROLE: test-only — no training, no 05_predict/ of its own. Evaluates models
# from other datasets (currently chaos). Becomes "both" if native training is added.
DATASET_ROLE="test-only"

# ── Cross-dataset model source: chaos ────────────────────────────────────────
# SLIVER07 consumes chaos checkpoints. nnUNetv2_predict resolves the model from
# nnUNet_results + the dataset id (the trained model dir holds plans.json/dataset.json),
# so predict points nnUNet_results at the chaos run dir. Set BEFORE common_env so
# CE_EXTRA_PYTHONPATH can reference CHAOS_SCRIPTS_DIR; ${VAR:-default} forms let
# sliver07's env_t2spir.sh pre-export the T2spir chaos vars.
export CHAOS_DATASET_ROOT="${DATASET_ROOT}/../chaos"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw"
export CHAOS_DATASET_ID="${CHAOS_DATASET_ID:-60}"
export CHAOS_DS_NAME="${CHAOS_DS_NAME:-Dataset060_CHAOS_MR_T1in}"
# chaos dataset.json (label map: background 0, liver 1, …) — used by 06_evaluate to
# score the liver label only (SLIVER07 GT annotates the liver alone).
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"
export CHAOS_TRAINING_CONTRAST="${CHAOS_TRAINING_CONTRAST:-t1in}"
export CHAOS_MODEL_TYPE="chaos_model"
# chaos scripts dir on PYTHONPATH so chaos trainer classes (nnUNetTrainerCHAOS*)
# resolve for -tr at predict time.
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="sliver07-liver"             # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw"                          # only 0_raw exists (no preprocessed/splits — eval-only)
CE_EXTRA_PYTHONPATH="${CHAOS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
