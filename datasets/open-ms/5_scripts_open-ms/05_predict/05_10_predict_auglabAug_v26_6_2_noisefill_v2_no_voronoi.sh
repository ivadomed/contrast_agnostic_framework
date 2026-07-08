#!/usr/bin/env bash
# Predict with the Voronoi-ablation arm (auglabAug_v26_6_2_noisefill_v2_no_voronoi_train025_val100)
# on the open-ms held-out test set (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_09).
# Same trainer class as OUR method (05_03) — only the training-time AugLab transform
# config differed (full noise fill + skip_sub_parc_prob=1.0, i.e. K-means-only
# partition, no Voronoi sub-parcellation); inference is identical.
# Usage: bash 05_10_predict_auglabAug_v26_6_2_noisefill_v2_no_voronoi.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_noisefill_v2_no_voronoi_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
# Trained on folds 0-2 only (TRAIN_FOLDS in 04_09) — no fold-3 checkpoint exists.
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
