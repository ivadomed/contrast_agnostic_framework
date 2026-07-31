#!/usr/bin/env bash
# Predict on MSD-SPLEEN CT with the CHAOS T1in auglabAug v26_6_2 @50% (train050_val100,
# OURS) model — the DualVal mirror needed for checkpoint_best eval alongside val000
# (see CLAUDE.md's DualVal trainer note). Uses the ValSynth trainer (val-time synth).
# MSD-SPLEEN has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_09_predict_chaos_auglabAug_v26_6_2_train050_val100.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
