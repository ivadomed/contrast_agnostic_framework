#!/usr/bin/env bash
# Source this file at the top of every lld-mmri-hcc pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_lld-mmri-hcc root)
#
# LLD-MMRI-HCC is an EVALUATION-ONLY dataset (see datasets/lld-mmri-hcc/README.md). We
# never train here -- we run models trained on `atlas-liver-hcc` (T1w CE-MRI) over 157
# hepatocellular-carcinoma patients from LLD-MMRI (Lou et al. 2025 / MedSAM2-annotated,
# T2WI + DWI phases) to measure cross-contrast generalization on the tumour/lesion
# label -- the actual research question atlas-liver-hcc's single-modality nature can't
# answer on its own. Hence there is NO 01_create_splits / 03_preprocess / 04_train
# stage (no preprocessed/splits subdirs); the ATLAS_* vars point predict/evaluate at
# the atlas-liver-hcc checkpoints.
#
# This is the lld-mmri-hcc "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="lld-mmri-hcc"
# DATASET_ROLE: test-only -- no training, no native models. Evaluates models from
# atlas-liver-hcc only.
DATASET_ROLE="test-only"

# ── Cross-dataset model source: atlas-liver-hcc ─────────────────────────────
# LLD-MMRI-HCC consumes atlas-liver-hcc checkpoints. nnUNetv2_predict resolves the
# model from nnUNet_results + the dataset id (the trained model dir holds
# plans.json/dataset.json), so predict points nnUNet_results at the atlas-liver-hcc
# run dir.
export ATLAS_DATASET_ROOT="${ATLAS_DATASET_ROOT:-${DATASET_ROOT}/../atlas-liver-hcc}"
export ATLAS_PREDICTIONS_ROOT="${ATLAS_PREDICTIONS_ROOT:-${ATLAS_DATASET_ROOT}/8_results_atlas-liver-hcc/01_predictions}"
export ATLAS_NNUNET_RAW="${ATLAS_NNUNET_RAW:-${ATLAS_DATASET_ROOT}/2_nnUNet_atlas-liver-hcc/raw}"
export ATLAS_NNUNET_PREPROCESSED="${ATLAS_NNUNET_PREPROCESSED:-${ATLAS_DATASET_ROOT}/2_nnUNet_atlas-liver-hcc/preprocessed}"
export ATLAS_DATASET_ID="${ATLAS_DATASET_ID:-80}"
export ATLAS_DS_NAME="${ATLAS_DS_NAME:-Dataset080_AtlasLiverHCC}"
# atlas-liver-hcc dataset.json (label map: background 0, liver 1, tumour 2) -- LLD-MMRI-HCC
# GT annotates the lesion alone as class 1, so evaluate uses --label_map '{"tumour": [2, 1]}'
# to remap the model's tumour class (2) onto this dataset's own GT numbering (1).
export ATLAS_DATASET_JSON="${ATLAS_NNUNET_RAW}/${ATLAS_DS_NAME}/dataset.json"
export ATLAS_TRAINING_CONTRAST="${ATLAS_TRAINING_CONTRAST:-t1w}"
export ATLAS_MODEL_TYPE="atlas_liver_hcc_model"
# atlas-liver-hcc scripts dir on PYTHONPATH so its trainer classes
# (nnUNetTrainerAtlasLiverHCC*) resolve for -tr at predict time.
ATLAS_SCRIPTS_DIR="${ATLAS_DATASET_ROOT}/5_scripts_atlas-liver-hcc"

# common_env config (plain vars -- consumed by common_env, not exported to the env):
BIDS_SUBDIR="lld-mmri-hcc"                # -> BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw"                          # only 0_raw exists (no preprocessed/splits -- eval-only)
CE_EXTRA_PYTHONPATH="${ATLAS_SCRIPTS_DIR}"
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"
