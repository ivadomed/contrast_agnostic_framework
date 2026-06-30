#!/usr/bin/env bash
# Source this file at the top of every trusted pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_trusted root)
#
# TRUSTED is an EVALUATION-ONLY dataset (see datasets/trusted/README.md). We never
# train here — we run models trained on `chaos` (MR T1-DUAL in-phase / T2-SPIR) over
# TRUSTED's kidney volumes to measure MR→{CT,US} domain-randomization generalization.
# Unlike sliver07 (CT only), TRUSTED has TWO test modalities: abdominal CT (48 vols,
# both kidneys) and 3D ultrasound (59 vols, one kidney each). Both are scored against
# a single binary kidney GT, so chaos's right_kidney(2)+left_kidney(3) predictions are
# MERGED into one "kidney" foreground at evaluate time (see 06_00_evaluate_trusted.py).
# Hence there is NO 01_create_splits / 04_train stage; the CHAOS_* vars point
# predict/evaluate at the chaos checkpoints.
#
# This is the trusted "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="trusted"
# DATASET_ROLE: test-only — no training, no native models. Evaluates models from
# other datasets (currently chaos). Becomes "both" if native training is added.
DATASET_ROLE="test-only"

# ── Cross-dataset model source: chaos ────────────────────────────────────────
# TRUSTED consumes chaos checkpoints. nnUNetv2_predict resolves the model from
# nnUNet_results + the dataset id (the trained model dir holds plans.json/dataset.json),
# so predict points nnUNet_results at the chaos run dir. Set BEFORE common_env so
# CE_EXTRA_PYTHONPATH can reference CHAOS_SCRIPTS_DIR; ${VAR:-default} forms let
# trusted's env_t2spir.sh pre-export the T2spir chaos vars.
export CHAOS_DATASET_ROOT="${DATASET_ROOT}/../chaos"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw"
export CHAOS_DATASET_ID="${CHAOS_DATASET_ID:-60}"
export CHAOS_DS_NAME="${CHAOS_DS_NAME:-Dataset060_CHAOS_MR_T1in}"
# chaos dataset.json (label map: background 0, liver 1, right_kidney 2, left_kidney 3,
# spleen 4). TRUSTED's evaluator (06_00_evaluate_trusted.py) hard-codes the kidney
# merge {2,3}→kidney itself, so this is exported for parity/debugging but the eval
# uses an explicit triple, not --labels on this json.
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"
export CHAOS_TRAINING_CONTRAST="${CHAOS_TRAINING_CONTRAST:-t1in}"
export CHAOS_MODEL_TYPE="chaos_model"
# chaos scripts dir on PYTHONPATH so chaos trainer classes (nnUNetTrainerCHAOS*)
# resolve for -tr at predict time.
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="trusted-kidney"             # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw"                          # only 0_raw exists (no preprocessed/splits — eval-only)
CE_EXTRA_PYTHONPATH="${CHAOS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
