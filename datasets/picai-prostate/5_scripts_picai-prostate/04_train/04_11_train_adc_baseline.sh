#!/usr/bin/env bash
# Train the real-data BASELINE (no synthesis, standard nnUNet augmentation) on picai-prostate ADC (Dataset081_PICAI_ADC).
# csPCa lesion segmentation; 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_11_train_adc_baseline.sh                                  # auto RUN_ID
#   bash 04_11_train_adc_baseline.sh picai-prostate_adc_baseline_<TS>   # resume
source "$(dirname "$0")/../00_utils/env_adc.sh"

METHOD="baseline"
TRAINER="nnUNetTrainerPICAIProstateBaseline"
CATEGORY="nnUNet"
DATASET_ID="081"
DA_WORKERS=6
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

# nnUNet-category method → 04_00_common.sh derives NNUNET_RESULTS_BASE
# as .../<contrast>/nnUNet from CATEGORY above.

source "$(dirname "$0")/04_00_common.sh" "$@"
