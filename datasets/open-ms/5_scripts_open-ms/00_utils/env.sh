#!/usr/bin/env bash
# Source this at the top of every open-ms pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_open-ms root)
#
# open-ms = brain MS lesion segmentation (open_ms_data / Lesjak 2018, CC-BY). The model
# is trained on FLAIR with synthesis augmentation and tested cross-contrast on T2W / T1W
# (all co-registered to FLAIR) — the same "train one contrast, generalise to others"
# setup as CHAOS (t1in → t1out/t2spir/ct) and BraTS (t1n → t2w/t1c/t2f).
#
# STANDARD results layout (as chaos / on-harmony): trained models are co-located with
# their predictions under
#   01_predictions/<MODEL_TYPE>/<TRAINING_CONTRAST>/<nnUNet|auglab>/<RUN_ID>/
# metrics under 02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/. nnUNet_results points at the
# nnUNet-category base; auglab train scripts override NNUNET_RESULTS_BASE → .../auglab.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="open-ms"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
export NNUNET_DATASET_ID="Dataset070_OpenMS_FLAIR"
export MODEL_TYPE="open_ms_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-flair}"
# 60h walltime: 2000 epochs is ~30h at ~55s/epoch, so 60h is ~2x headroom AND short
# enough that Slurm backfill can slot folds into gaps (much faster to get an L40S than a
# 167h job). If a fold ever hits the wall, resume with the same RUN_ID (--c, auto from
# checkpoint_latest). Raise only if epochs run far slower than expected.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-60:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# BIDS tree built by 00_utils/00_01_bidsify.py (0_raw → 1_BIDS, hard-linked); 02_00_convert.py
# reads from here. Set BIDS_ROOT explicitly to the named leaf (like chaos-abdominal).
export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_open-ms/open-ms-brain"
export METRICS_ROOT="${DATASET_ROOT}/8_results_open-ms/02_metrics"
export nnUNet_results="${DATASET_ROOT}/8_results_open-ms/01_predictions/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_open-ms"
export RESULTS_DIR="${DATASET_ROOT}/8_results_open-ms"
