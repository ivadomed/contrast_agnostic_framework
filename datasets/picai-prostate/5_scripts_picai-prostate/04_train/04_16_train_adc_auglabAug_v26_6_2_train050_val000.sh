#!/usr/bin/env bash
# Train OUR METHOD (AugLab default augs + V26_6_2 GPU contrast synthesis @50% train), DUAL VALIDATION on picai-prostate ADC (Dataset081_PICAI_ADC).
# csPCa lesion segmentation; 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# DUAL VALIDATION — this ONE run yields BOTH headline arms of our method:
#   checkpoint_best.pth        -> clean/val000 best  (this RUN_ID, named _val000_)
#   checkpoint_best_val100.pth -> synth-only/val100 best
# At on_train_end the trainer hard-links a sibling RUN_ID with "_val000_" swapped for
# "_val100_", so BOTH show up as ordinary predict-ready runs (see
# picai_prostate/trainers/auglab_dualval.py). RUN_ID MUST be named "_val000_" — METHOD
# below ensures that. PREDICT NOTE: the predict wrappers for BOTH RUN_IDs must set
# TRAINER=nnUNetTrainerPICAIProstateAugLabDualVal (both mirrors live under this trainer's
# directory name — NOT AugLabDefault/AugLabValSynth).
#
# Usage:
#   bash 04_16_train_adc_auglabAug_v26_6_2_train050_val000.sh                                  # auto RUN_ID
#   bash 04_16_train_adc_auglabAug_v26_6_2_train050_val000.sh picai-prostate_adc_auglabAug_v26_6_2_train050_val000_<TS>   # resume
source "$(dirname "$0")/../00_utils/env_adc.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerPICAIProstateAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="081"
DA_WORKERS=6
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

source "$(dirname "$0")/04_00_common.sh" "$@"
