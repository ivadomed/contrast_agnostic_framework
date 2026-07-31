#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS T1in auglabAug v26_6_2 @50% (train050_val100)
# model — the val000/val100 significance comparison's cross-dataset arm. Uses chaos
# Dataset060_CHAOS_MR_T1in checkpoints (t1in is the env.sh default CHAOS contrast, so
# no CHAOS_* override needed). train050_val100 uses the ValSynth trainer (val-time
# synth), unlike train050_val000's plain V26_6_2 trainer.
# Usage: bash 05_20_predict_chaos_t1in_auglabAug_v26_6_2_train050_val100.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
