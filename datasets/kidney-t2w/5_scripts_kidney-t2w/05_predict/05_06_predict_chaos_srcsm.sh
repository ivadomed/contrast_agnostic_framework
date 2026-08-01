#!/usr/bin/env bash
# Predict on KIDNEY-T2W with the CHAOS-TRAINED srcsm (SRCSM SemRandConv-3D) model.
# KIDNEY-T2W has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_06_predict_chaos_srcsm.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_srcsm_20260710_011817}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
