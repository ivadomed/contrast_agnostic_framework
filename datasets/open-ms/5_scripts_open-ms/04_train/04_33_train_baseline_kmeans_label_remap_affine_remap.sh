#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, extra arm: v26_6_2 ALONE minus Voronoi —
# same ImageContrastV26_6_2GPUTransform class as 04_26/04_06 (v26_6_2 alone
# headline, the REAL (mu, alpha) affine remap), NOT the NoiseFill-ablation
# class used by 04_24/04_25/04_10/04_11 (label_remap/voronoi rungs) — just
# skip_sub_parc_prob=1.0 instead of the headline's 0.4, i.e. Voronoi
# sub-parcellation turned off. Isolates Voronoi's contribution on the REAL
# mechanism. Extra arm — not a replacement for the standard 6-method suite.
#
# Uses nnUNetTrainerOpenMSAugLabDualVal (ported from CHAOS's DualVal trainer,
# see datasets/open-ms/5_scripts_open-ms/open_ms/trainers/auglab_dualval.py):
# ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# instead of training this config twice (open-ms's older ladder rungs each
# trained val000/val100 as separate runs — this is the first open-ms rung to
# use the single-run DualVal approach).
#
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. nnUNet-category (matches 04_26's placement).
#
# Usage:
#   bash 04_33_train_baseline_kmeans_label_remap_affine_remap.sh                                        # auto RUN_ID
#   bash 04_33_train_baseline_kmeans_label_remap_affine_remap.sh open-ms_flair_baseline_kmeans_label_remap_affine_remap_train050_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_affine_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDualVal"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline_kmeans_label_remap_affine_remap_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_affine_remap_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform_no_voronoi.json"

# nnUNet-category model -> uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_26. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
