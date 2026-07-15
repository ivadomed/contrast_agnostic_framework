#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, extra arm: v26_6_2 ALONE minus Voronoi, @50%
# train, 100% val synth. Same ImageContrastV26_6_2GPUTransform class as 04_06
# (v26_6_2 alone headline, the REAL (mu, alpha) affine remap), NOT the
# NoiseFill-ablation class used by 04_10/04_11 (label_remap/voronoi rungs) —
# just skip_sub_parc_prob=1.0 instead of 04_06's 0.4, i.e. Voronoi
# sub-parcellation turned off. Isolates Voronoi's contribution on the REAL
# mechanism. Companion to 04_33 (val000 variant). Extra arm — not a
# replacement for the standard 6-method suite.
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category (matches 04_06's placement).
#
# Usage:
#   bash 04_34_train_baseline_kmeans_label_remap_affine_remap_train050_val100.sh                                        # auto RUN_ID
#   bash 04_34_train_baseline_kmeans_label_remap_affine_remap_train050_val100.sh open-ms_flair_baseline_kmeans_label_remap_affine_remap_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_affine_remap_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline_kmeans_label_remap_affine_remap_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_affine_remap_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform_no_voronoi.json"

# nnUNet-category model -> uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_06 (v26_6_2 alone). No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
