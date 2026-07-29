#!/usr/bin/env bash
# Override OPENMS_* env vars for T1w-TRAINED open-ms models before sourcing env.sh.
# Source this instead of env.sh when predicting/evaluating with open-ms's T1w-trained
# model family (Dataset071_OpenMS_T1W) rather than its default FLAIR-trained one
# (Dataset070_OpenMS_FLAIR). Mirrors trusted's env_t2spir.sh (the second-contrast
# pattern) / open-ms's own env_t1w.sh.
#
# Usage: source env_t1w.sh   (from a wrapper that would otherwise source env.sh)
#
# Must be sourced BEFORE env.sh (wrappers that source this need NOT source env.sh
# separately; this script sources it for them via the pre-export mechanism).
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
# env.sh uses ${VAR:-default} guards (see common_env.sh's fix, 2026-07-29) so these
# pre-exports are preserved through the re-source predict_common.sh's wrapper chain does.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
