#!/usr/bin/env bash
# Predict on KIDNEY-T2W with the CHAOS-TRAINED SynthSeg (EM) model.
# KIDNEY-T2W has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_03_predict_chaos_synthseg_EM.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_synthseg_EM_train100_val000_20260611_120000}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
