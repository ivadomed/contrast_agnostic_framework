#!/usr/bin/env bash
# Predict with the LEVEL-3 CAUSAL ABLATION (auglabAug_v26_6_2_noisefill_train025_val100)
# on the open-ms held-out test set (FLAIR/T2W/T1W, all folds). Same trainer class as OUR
# method (05_03) — only the training-time AugLab transform config differed (Gaussian
# noise-fill vs real-intensity remap); inference is identical.
# Usage: bash 05_08_predict_auglabAug_v26_6_2_noisefill.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_noisefill_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
