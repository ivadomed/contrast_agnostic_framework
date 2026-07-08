#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, rung 3 of 4 — K-means + Voronoi + noise fill, ALONE
# (no full AugLab intensity suite — only nnU-Net's own spatial DA, matching 04_06's
# "v26_6_2 alone" recipe exactly, just swapping the fill to noise). Identical to 04_10
# except skip_sub_parc_prob=0.4 (Voronoi sub-parcellation active).
# Ladder: baseline (04_01, nothing) -> +K-means/noise (04_10) -> THIS -> v26_6_2 alone
# (04_06, PALETTE alone, real texture). Complements the auglab-anchored ladder
# (04_08/04_09/04_03).
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category (matches 04_06's placement).
#
# Usage:
#   bash 04_11_train_v26_6_2_noisefill_v2_alone_train050.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_noisefill_v2_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_v26_6_2_noisefill_v2_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
export TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2}"
export RUN_JOB_TIME_DEFAULT=24:00:00

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_noisefill_v2_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2NoiseFillV2GPUTransform.json"

# nnUNet-category model → uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_06 (v26_6_2 alone). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
