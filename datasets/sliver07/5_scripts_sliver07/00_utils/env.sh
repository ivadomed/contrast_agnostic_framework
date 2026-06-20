#!/usr/bin/env bash
# Source this file at the top of every sliver07 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_sliver07 root)
#
# SLIVER07 is an EVALUATION-ONLY dataset (see datasets/sliver07/README.md). We never
# train here — we run models trained on `chaos` (MR T1-DUAL in-phase) over SLIVER07's
# 20 labeled CT volumes to measure MR→CT domain-randomization generalization.
# Hence there is NO 01_create_splits / 03_preprocess / 04_train stage; the CHAOS_*
# vars below point the predict/evaluate steps at the chaos-trained checkpoints.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

export RAW_ROOT="${DATASET_ROOT}/0_raw_sliver07"
export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_sliver07/sliver07-liver"
# nnUNet inference inputs (built by 05_00_build_test_inputs.py). No training data here.
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_sliver07/raw"
# Prediction output root: 01_predictions/{category}/{run_id}/fold{k}/
export PREDICTIONS_ROOT="${DATASET_ROOT}/8_results_sliver07/01_predictions"
# Metrics output root: 02_metrics/{category}_{run_id}/fold{k}/
export METRICS_ROOT="${DATASET_ROOT}/8_results_sliver07/02_metrics"

export DATASET_NAME="sliver07"
# DATASET_ROLE: test-only — no training, no 05_predict/. Evaluates models from
# other datasets (currently chaos) for cross-dataset generalization benchmarking.
# When AMOS or similar is added, this will become "both".
export DATASET_ROLE="test-only"

# ── Cross-dataset model source: chaos ────────────────────────────────────────
# SLIVER07 consumes chaos checkpoints. nnUNetv2_predict resolves the model from
# nnUNet_results + the dataset id (the trained model dir holds plans.json /
# dataset.json), so predict points nnUNet_results at the chaos run dir below.
export CHAOS_DATASET_ROOT="${DATASET_ROOT}/../chaos"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw"
export CHAOS_DATASET_ID="60"
export CHAOS_DS_NAME="Dataset060_CHAOS_MR_T1in"
# chaos dataset.json (label map: background 0, liver 1, …) — used by 06_evaluate to
# score the liver label only (SLIVER07 GT annotates the liver alone).
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"
# Training contrast and model type for chaos (used in result path structure).
export CHAOS_TRAINING_CONTRAST="t1in"
export CHAOS_MODEL_TYPE="chaos_model"

# Make the sliver07 scripts dir importable, plus the chaos scripts dir so the chaos
# trainer classes (nnUNetTrainerCHAOS*) resolve for -tr at predict time.
SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_sliver07"
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"
export PYTHONPATH="${SCRIPTS_DIR}:${CHAOS_SCRIPTS_DIR}:${PYTHONPATH:-}"
