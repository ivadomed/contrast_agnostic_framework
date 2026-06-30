#!/usr/bin/env bash
# Predict on TRUSTED (CT + US) with the CHAOS-TRAINED synthseg_noEM model (contrast: t1in).
# TRUSTED has no model of its own — see 05_01_predict_common.sh. Default items: ct us.
# Usage: bash 05_06_predict_chaos_synthseg_noEM.sh [CHAOS_RUN_ID] [FOLD] [ITEM ...]   (ITEM ∈ {ct,us})
set -euo pipefail
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_synthseg_noEM_train100_val000_20260611_120000}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
