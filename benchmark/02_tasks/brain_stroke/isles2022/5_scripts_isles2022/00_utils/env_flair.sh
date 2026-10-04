#!/usr/bin/env bash
# isles2022 FLAIR-training context: pre-exports TRAINING_CONTRAST + nnUNet_results, then sources env.sh.
export TRAINING_CONTRAST="flair"
export nnUNet_results="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/8_results_isles2022/01_predictions/isles2022_model/flair/nnUNet"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
