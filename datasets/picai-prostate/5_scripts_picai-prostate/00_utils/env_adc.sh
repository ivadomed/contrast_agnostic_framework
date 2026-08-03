#!/usr/bin/env bash
# ADC training environment for picai-prostate — source this INSTEAD of env.sh for ADC
# experiments. Sources env.sh first (all shared paths), then overrides the
# contrast-specific vars. Mirrors open-ms's env_t1w.sh / chaos's env_t2spir.sh
# (the second-training-modality pattern).
#
# The model is trained on ADC with synthesis augmentation and tested cross-contrast on
# T2W / ADC / HBV (all resampled onto one common grid by 00_01_bidsify.py, so one lesion
# mask serves every contrast) — the "train one contrast, generalise to others" setup.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_adc.sh"

# Pre-export the contrast-specific vars BEFORE sourcing env.sh so they survive the
# `:-` defaults there (and the re-source that 04_00_common.sh does).
export TRAINING_CONTRAST="adc"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# env.sh rebuilds nnUNet_results from TRAINING_CONTRAST=adc → .../adc/nnUNet already.
# Only NNUNET_DATASET_ID must be overridden to the ADC nnUNet dataset (env.sh hardcodes
# the T2W one). Note: training keys off the numeric DATASET_ID (081) set in each wrapper,
# not this var — this keeps convert/preprocess pointed at the right dataset.
export NNUNET_DATASET_ID="Dataset081_PICAI_ADC"
