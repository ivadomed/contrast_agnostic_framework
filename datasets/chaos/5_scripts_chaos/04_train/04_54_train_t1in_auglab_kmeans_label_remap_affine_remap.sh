#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, extra arm: auglabAug_v26_6_2 (OURS) minus
# Voronoi (K-means + per-anatomical-label remap + the REAL (mu, alpha) affine
# remap — same ImageContrastV26_6_2GPUTransform class as the auglabAug_v26_6_2
# headline, NOT the NoiseFill-ablation class used by the other
# auglab_kmeans/label_remap/voronoi rungs — just skip_sub_parc_prob=1.0
# instead of the headline's 0.4, i.e. Voronoi sub-parcellation turned off).
# Isolates Voronoi's contribution on the REAL mechanism, on top of the full
# AugLab intensity stack. Per CLAUDE.md this is an EXTRA arm, not a
# replacement for the standard 6-method suite.
#
# Uses nnUNetTrainerCHAOSAugLabDualVal: ONE training run produces BOTH
#   checkpoint_best.pth         (clean/val000 best)
#   checkpoint_best_val100.pth  (synth-only/val100 best)
# See datasets/chaos/5_scripts_chaos/chaos/trainers/auglab_dualval.py.
#
# 3 folds (0 1 2), 1 GPU per fold, 200 epochs (CHAOS standard).
#
# Usage:
#   bash 04_54_train_t1in_auglab_kmeans_label_remap_affine_remap.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_label_remap_affine_remap_train025_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
DATASET_ID="060"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_t1in_auglab_kmeans_label_remap_affine_remap_train025_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_kmeans_label_remap_affine_remap_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform_no_voronoi.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
