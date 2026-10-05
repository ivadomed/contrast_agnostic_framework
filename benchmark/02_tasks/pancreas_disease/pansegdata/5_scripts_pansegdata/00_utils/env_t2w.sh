#!/usr/bin/env bash
# pansegdata T2W-training context: pre-exports TRAINING_CONTRAST, then sources env.sh (which derives nnUNet_results from it).
export TRAINING_CONTRAST="t2w"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
