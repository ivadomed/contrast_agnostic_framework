#!/usr/bin/env bash
# Predict with the CT-trained OURS (auglabAug_v26_6_2_train050) val100 (synth-validated)
# checkpoint mirror — the sibling RUN_ID the DualVal trainer materializes at
# on_train_end() (same fold data, different best-checkpoint selection). Only needed for
# a best-checkpoint comparison; checkpoint_final.pth is identical between the two
# mirrors (see CLAUDE.md's DualVal note), so a final-checkpoint eval only needs val000.
# Usage: bash 05_08_predict_ct_ours_val100.sh <RUN_ID_with__val100_> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAutoPETAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="120"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
