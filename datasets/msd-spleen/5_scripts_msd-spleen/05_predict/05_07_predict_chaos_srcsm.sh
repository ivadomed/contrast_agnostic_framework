#!/usr/bin/env bash
# Predict on MSD-SPLEEN CT with the CHAOS-TRAINED srcsm model (SRCSM SemRandConv-3D,
# Thaler et al. 2025 — now a fixed member of the 6-method suite). MSD-SPLEEN has no
# model of its own — see 05_01_predict_common.sh. srcsm reuses the AugLab-default
# trainer (auglab category).
# Usage: bash 05_07_predict_chaos_srcsm.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_srcsm_20260710_011817}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
