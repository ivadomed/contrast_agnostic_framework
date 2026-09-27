#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS T1in auglabAug v26_6_2 @50% (train050_val000)
# model — the extra ablation arm (cross-dataset probe). Uses chaos Dataset060_CHAOS_MR_T1in
# checkpoints (t1in is the env.sh default CHAOS contrast, so no CHAOS_* override needed).
# train050_val000 uses the plain V26_6_2 trainer (no val-time synth). 4-fold run →
# prediction defaults to folds 0 1 2.
# Usage: bash 05_19_predict_chaos_t1in_auglabAug_v26_6_2_train050_val000.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
