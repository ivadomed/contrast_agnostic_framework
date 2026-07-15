#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, extra arm: auglabAug_v26_6_2 (OURS) minus
# Voronoi, @50% train (bumped from the branch's usual 25% to match 04_22's
# val000 convention), 0% val synth (clean validation) — same
# ImageContrastV26_6_2GPUTransform class as 04_22 (auglabAug_v26_6_2 headline,
# the REAL (mu, alpha) affine remap), NOT the NoiseFill-ablation class used by
# 04_27/04_28 (label_remap/voronoi rungs) — just skip_sub_parc_prob=1.0
# instead of 04_22's 0.4, i.e. Voronoi sub-parcellation turned off. Isolates
# Voronoi's contribution on the REAL mechanism, on top of the full AugLab
# intensity stack. Companion to 04_36 (val100 variant, train025). Extra arm —
# not a replacement for the standard 6-method suite.
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. AugLabDefault trainer.
#
# Usage:
#   bash 04_35_train_auglab_kmeans_label_remap_affine_remap_train050_val000.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_label_remap_affine_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglab_kmeans_label_remap_affine_remap_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_kmeans_label_remap_affine_remap_train050.json"

# AugLabDefault trainer -> clean (real-image) validation, no AUGLAB_VAL_PARAMS_GPU_JSON needed.
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
