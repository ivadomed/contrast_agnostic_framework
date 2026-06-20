#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS-TRAINED SynthSeg (noEM) model.
# SLIVER07 has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_06_predict_chaos_synthseg_noEM.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_synthseg_noEM_train100_val000_20260611_120000}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
