#!/usr/bin/env bash
# T2spir training environment — source this instead of env.sh for T2spir experiments.
# Sources env.sh first (all shared paths), then overrides the contrast-specific vars.
#
# Usage (from a step subdir):
#   source "$(dirname "$0")/../00_utils/env_t2spir.sh"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

export TRAINING_CONTRAST="t2spir"
export NNUNET_DATASET_ID="Dataset061_CHAOS_MR_T2spir"
# Override the default nnUNet_results path (env.sh hardcodes t1in).
export nnUNet_results="${DATASET_ROOT}/8_results_chaos/01_predictions/chaos_model/t2spir/nnUNet"
