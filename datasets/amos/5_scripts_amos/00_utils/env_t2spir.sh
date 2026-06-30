#!/usr/bin/env bash
# Override CHAOS_* env vars for T2spir models before sourcing env.sh.
# Source this instead of env.sh when predicting with chaos T2spir-trained models.
# Usage: source env_t2spir.sh  (from a predict wrapper that pre-exports CHAOS_* vars)
#
# Must be sourced BEFORE env.sh (wrappers that source this need NOT source env.sh separately;
# this script sources it for them via the pre-export mechanism).
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
# env.sh uses ${VAR:-default} so these pre-exports are preserved.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
