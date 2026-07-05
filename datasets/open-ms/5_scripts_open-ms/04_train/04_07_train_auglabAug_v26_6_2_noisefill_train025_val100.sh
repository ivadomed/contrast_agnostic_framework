#!/usr/bin/env bash
# LEVEL-3 CAUSAL ABLATION — PALETTE partition + SynthSeg-style Gaussian fill, open-ms FLAIR.
# Identical to 04_03 (auglabAug_v26_6_2, our method) EXCEPT the V26_6_2 contrast transform's
# per-region fill is swapped from real-intensity remap → Gaussian noise (texture destroyed),
# holding the K-means+Voronoi partition fixed. Isolates texture preservation as the causal
# variable. Compare downstream against the already-trained PALETTE arm (04_03).
# 4 folds, 1 GPU/fold, 2000 epochs. AugLabValSynth trainer (synth-only validation).
#
# Usage:
#   bash 04_07_train_auglabAug_v26_6_2_noisefill_train025_val100.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_noisefill_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_open-ms_auglabAug_v26_6_2_noisefill_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2NoiseFillGPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2NoiseFillGPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
