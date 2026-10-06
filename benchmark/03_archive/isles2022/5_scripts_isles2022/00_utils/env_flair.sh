#!/usr/bin/env bash
# isles2022 FLAIR-training context: pre-exports TRAINING_CONTRAST, then sources env.sh (which derives nnUNet_results from it).
export TRAINING_CONTRAST="flair"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
