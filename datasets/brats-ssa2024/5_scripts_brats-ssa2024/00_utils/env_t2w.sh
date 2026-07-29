#!/usr/bin/env bash
# Override BRATS_* env vars for T2w-TRAINED brats2024-glioma models before sourcing
# env.sh. Source this instead of env.sh when predicting/evaluating with the T2w-trained
# model family (Dataset052_BraTS2024GliomaT2w) rather than the default T1n-trained one
# (Dataset051_BraTS2024GliomaT1n). Mirrors trusted's env_t2spir.sh / mslesseg's env_t1w.sh.
#
# Usage: source env_t2w.sh   (from a wrapper that would otherwise source env.sh)
#
# Must be sourced BEFORE env.sh (wrappers that source this need NOT source env.sh
# separately; this script sources it for them via the pre-export mechanism).
export BRATS_TRAINING_CONTRAST="t2w"
export BRATS_DATASET_ID="052"
export BRATS_DS_NAME="Dataset052_BraTS2024GliomaT2w"
# env.sh uses ${VAR:-default} guards so these pre-exports are preserved through the
# re-source predict_common.sh's wrapper chain does.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
