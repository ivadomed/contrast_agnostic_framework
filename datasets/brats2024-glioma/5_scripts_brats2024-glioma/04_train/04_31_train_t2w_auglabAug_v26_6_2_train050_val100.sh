#!/usr/bin/env bash
# Train OUR METHOD: AugLab default augs + V26_6_2 GPU contrast synthesis @50% train,
# 100% val synth, on BraTS 2024 Glioma T2w. AugLabValSynth trainer (synth-only validation).
# Matches auglabAug_v26_6_2_train050_val000 (train fraction) but with synth-only val,
# closing the matched train050 val000/val100 pair used for the deployed-config ladder.
# 3 folds, 1 GPU/fold, 2500 epochs.
#
# Usage:
#   bash 04_31_train_t2w_auglabAug_v26_6_2_train050_val100.sh                                        # auto RUN_ID
#   bash 04_31_train_t2w_auglabAug_v26_6_2_train050_val100.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs x ~60s/ep ~= 42h
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth"
DATASET_ID="052"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2w_auglabAug_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
