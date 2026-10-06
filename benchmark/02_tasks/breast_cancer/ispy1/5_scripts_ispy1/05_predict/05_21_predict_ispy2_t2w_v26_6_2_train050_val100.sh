#!/usr/bin/env bash
# Predict on ispy1 with the ispy2 t2w causal-ablation ladder model (v26_6_2_train050_val100).
# Usage: bash 05_21_predict_ispy2_t2w_v26_6_2_train050_val100.sh [RUN_ID] [FOLD]
set -euo pipefail
# RETIRED 2026-10-06: the default RUN_ID below is the ValSynth rung-5 run (checkpoint_best chosen on SYNTHETIC
# validation). The val000 rung 5 is predicted by the rung-4 predict wrapper (05_16/05_20) with its own RUN_ID
# (scripts/cluster/rung5_val000/post_run.sh). Set ALLOW_VAL100_ALONE=1 only to re-predict the old val100 run.
[ "${ALLOW_VAL100_ALONE:-0}" = 1 ] || { echo "ERROR: $(basename "${BASH_SOURCE[0]}") is retired (val100 rung 5); use the rung-4 predict wrapper with the val000 RUN_ID" >&2; exit 1; }
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
CATEGORY="nnUNet"
export ISPY2_TRAINING_CONTRAST="t2w"
export ISPY2_DATASET_ID="101"
RUN_ID="${1:-ispy2_t2w_v26_6_2_train050_val100_20260905_163655}"
source "$(dirname "$0")/05_01_predict_ispy2_common.sh" "$RUN_ID" "${@:2}"
