#!/usr/bin/env bash
# exp2: AugLab + SynthSeg_EM augmentation on CHAOS, fold 0 only, 300 epochs.
# Uses the new config with auglab augmentations actually enabled (prob > 0).
# Runs on GPU 2 (SINGLE_GPU=2).
#
# Usage:
#   bash 04_08_train_exp2_auglabAug_synthseg_EM.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_synthseg_EM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-300}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_Synthseg_EM.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

export SINGLE_FOLD="${SINGLE_FOLD:-0}"
export SINGLE_SLOT="${SINGLE_SLOT:-2}"
export SINGLE_GPU="${SINGLE_GPU:-2}"

source "$(dirname "$0")/04_00_common.sh" "$@"
