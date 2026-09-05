#!/usr/bin/env bash
# Source this file at the top of every liverhccseg pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_liverhccseg root)
#
# LIVERHCCSEG is an EVALUATION-ONLY dataset (see datasets/liverhccseg/README.md). We
# never train here -- we run models trained on `atlas-liver-hcc` (T1w CE-MRI) over 14
# HCC patients from LiverHccSeg (TCGA-LIHC, real expert masks -- 2 independent
# board-certified radiologist raters, rater1 used as active GT) to add an independent,
# same-disease, same-CE-T1w-family-but-different-cohort/scanner stratum alongside
# lld-mmri-hcc's genuinely cross-contrast (T2WI/DWI) one. See
# datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/configs/
# atlas-liver-hcc_cross_dataset_t1w_01_results.yaml for how both pool into one table.
# Hence there is NO 01_create_splits / 03_preprocess / 04_train stage; the ATLAS_* vars
# point predict/evaluate at the atlas-liver-hcc checkpoints.
#
# This is the liverhccseg "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="liverhccseg"
DATASET_ROLE="test-only"

# ── Cross-dataset model source: atlas-liver-hcc ─────────────────────────────
export ATLAS_DATASET_ROOT="${ATLAS_DATASET_ROOT:-${DATASET_ROOT}/../atlas-liver-hcc}"
export ATLAS_PREDICTIONS_ROOT="${ATLAS_PREDICTIONS_ROOT:-${ATLAS_DATASET_ROOT}/8_results_atlas-liver-hcc/01_predictions}"
export ATLAS_NNUNET_RAW="${ATLAS_NNUNET_RAW:-${ATLAS_DATASET_ROOT}/2_nnUNet_atlas-liver-hcc/raw}"
export ATLAS_NNUNET_PREPROCESSED="${ATLAS_NNUNET_PREPROCESSED:-${ATLAS_DATASET_ROOT}/2_nnUNet_atlas-liver-hcc/preprocessed}"
export ATLAS_DATASET_ID="${ATLAS_DATASET_ID:-80}"
export ATLAS_DS_NAME="${ATLAS_DS_NAME:-Dataset080_AtlasLiverHCC}"
# atlas-liver-hcc dataset.json (label map: background 0, liver 1, tumour 2) --
# LIVERHCCSEG has real liver masks too, but only TUMOUR is scored here (matching
# lld-mmri-hcc exactly), so every OOD/cross-dataset column in the pooled significance
# test measures the same construct. --label_map '{"tumour": [2, 1]}' (see
# 06_evaluate/06_01_evaluate_run.sh) -- same mechanism as lld-mmri-hcc.
export ATLAS_DATASET_JSON="${ATLAS_NNUNET_RAW}/${ATLAS_DS_NAME}/dataset.json"
export ATLAS_TRAINING_CONTRAST="${ATLAS_TRAINING_CONTRAST:-t1w}"
export ATLAS_MODEL_TYPE="atlas_liver_hcc_model"
ATLAS_SCRIPTS_DIR="${ATLAS_DATASET_ROOT}/5_scripts_atlas-liver-hcc"

# common_env config (plain vars -- consumed by common_env, not exported to the env):
BIDS_SUBDIR="liverhccseg"
CE_SUBDIRS="raw"
CE_EXTRA_PYTHONPATH="${ATLAS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
