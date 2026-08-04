#!/usr/bin/env bash
# Source this at the top of every atlas-liver-hcc pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_atlas-liver-hcc root)
#
# atlas-liver-hcc = ATLAS challenge HCC liver tumour segmentation on contrast-enhanced
# T1w MRI (Quinton et al. 2023, CC BY-NC-SA 4.0). Intra-tissue segmentation task (tumour
# vs. surrounding liver parenchyma — texture-defined, same category as open-ms/brats2024-
# glioma) but SINGLE MODALITY: unlike our other 4 training datasets, ATLAS provides only
# one training contrast (T1w CE-MRI, contrast phase varies per patient as metadata, not
# as a separate channel), so there is no second training contrast and no
# combined_contrasts pooling step for this dataset — the held-out test-patient split IS
# the generalization test.
#
# STANDARD results layout (as chaos / on-harmony / open-ms): trained models are
# co-located with their predictions under
#   01_predictions/<MODEL_TYPE>/<TRAINING_CONTRAST>/<nnUNet|auglab>/<RUN_ID>/
# metrics under 02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/. nnUNet_results points at the
# nnUNet-category base; auglab train scripts override NNUNET_RESULTS_BASE → .../auglab.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="atlas-liver-hcc"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
export NNUNET_DATASET_ID="Dataset080_AtlasLiverHCC"
export MODEL_TYPE="atlas_liver_hcc_model"
export TRAINING_CONTRAST="t1w"     # single modality, permanently — never overridden
# 2000-epoch horizon (same as open-ms/on-harmony). 60h walltime gives ~2x headroom over
# the ~30h open-ms took at 2000 epochs while staying short enough for Slurm backfill.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-60:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# No 1_BIDS step: ATLAS arrives pre-shaped as nnUNet-style imagesTr/labelsTr (see
# 02_nnunet/02_00_convert.py), so 02_00_convert.py reads directly from 0_raw — unlike
# open-ms/chaos there is no BIDS intermediate to build first.
export METRICS_ROOT="${DATASET_ROOT}/8_results_atlas-liver-hcc/02_metrics"
export nnUNet_results="${DATASET_ROOT}/8_results_atlas-liver-hcc/01_predictions/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_atlas-liver-hcc"
export RESULTS_DIR="${DATASET_ROOT}/8_results_atlas-liver-hcc"
