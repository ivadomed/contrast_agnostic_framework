#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, extra arm: auglabAug_v26_6_2 (OURS) minus
# Voronoi, @25% train, 100% val synth — same ImageContrastV26_6_2GPUTransform
# class as 04_03 (auglabAug_v26_6_2 headline, the REAL (mu, alpha) affine
# remap), NOT the NoiseFill-ablation class used by 04_08/04_09
# (label_remap/voronoi rungs) — just skip_sub_parc_prob=1.0 instead of
# 04_03's 0.4, i.e. Voronoi sub-parcellation turned off. Isolates Voronoi's
# contribution on the REAL mechanism, on top of the full AugLab intensity
# stack. Companion to 04_35 (val000 variant, train050). Extra arm — not a
# replacement for the standard 6-method suite.
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. AugLabValSynth trainer.
#
# Usage:
#   bash 04_36_train_auglab_kmeans_label_remap_affine_remap_train025_val100.sh                                        # auto RUN_ID
#   bash 04_36_train_auglab_kmeans_label_remap_affine_remap_train025_val100.sh open-ms_flair_auglab_kmeans_label_remap_affine_remap_train025_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_label_remap_affine_remap_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglab_kmeans_label_remap_affine_remap_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_kmeans_label_remap_affine_remap_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform_no_voronoi.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
