#!/usr/bin/env bash
# Override CHAOS_* env vars for T2spir models before sourcing env.sh.
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
