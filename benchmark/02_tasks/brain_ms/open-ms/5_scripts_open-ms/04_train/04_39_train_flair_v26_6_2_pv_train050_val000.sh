#!/usr/bin/env bash
# CAUSAL LADDER rung 6 (flair): rung 5 (04_26_train_v26_6_2_train050_val000.sh, PALETTE alone, real fill) +
# boundary partial-volume simulation (pv_levels in the train config). The ONLY diff vs
# 04_26_train_v26_6_2_train050_val000.sh is the train config (+ pv keys) and METHOD/LOG_DIR naming;
# trainer, val config, epochs unchanged. Generated from the rung-5 wrapper -- keep in sync.
# BASELINE-ANCHORED CAUSAL LADDER, rung 4 of 4 (v26_6_2 ALONE — real-texture fill,
# the "Ours, v26 only" row), @50% train, 0% val synth (clean validation). Same
# spatialDA-only config as 04_06, but AugLabDefault trainer instead of
# nnUNetTrainerOpenMSV26_6_2 so validation runs on real images only (mirrors how
# 04_22 switched auglabAug_v26_6_2 to AugLabDefault for its train050_val000 arm).
# Completes the baseline-anchored ladder @ train050_val000 alongside 04_24/04_25.
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category (matches 04_06's placement).
#
# Usage:
#   bash 04_39_train_flair_v26_6_2_pv_train050_val000.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_pv_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_v26_6_2_pv_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"

# AugLabDefault trainer -> clean (real-image) validation, no AUGLAB_VAL_PARAMS_GPU_JSON needed.
# nnUNet-category model -> uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_06. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
