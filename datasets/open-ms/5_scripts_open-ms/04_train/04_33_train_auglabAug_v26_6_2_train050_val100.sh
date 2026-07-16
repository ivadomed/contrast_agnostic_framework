#!/usr/bin/env bash
# Train OUR METHOD: AugLab default augs + V26_6_2 GPU contrast synthesis @50% train,
# 100% val synth, on open-ms FLAIR. AugLabValSynth trainer (synth-only validation).
# Matches auglabAug_v26_6_2_train050_val000 (train fraction) but with synth-only val,
# closing the matched train050 val000/val100 pair used for the deployed-config ladder.
# 3 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_33_train_auglabAug_v26_6_2_train050_val100.sh                                        # auto RUN_ID
#   bash 04_33_train_auglabAug_v26_6_2_train050_val100.sh open-ms_flair_auglabAug_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglabAug_v26_6_2_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
