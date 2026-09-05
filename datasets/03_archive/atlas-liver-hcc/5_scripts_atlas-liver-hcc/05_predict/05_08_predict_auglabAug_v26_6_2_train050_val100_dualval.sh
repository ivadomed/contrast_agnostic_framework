#!/usr/bin/env bash
# Predict the atlas-liver-hcc DUAL-VAL run's synth-only/val100 checkpoint — i.e. the
# "_val100_" MIRROR RUN_ID auto-materialized by 04_06's dual-val run at on_train_end.
# The mirror lives under the SAME nnUNetTrainerAtlasLiverHCCAugLabDualVal trainer dir as
# its val000 sibling, so predict MUST use TRAINER=...AugLabDualVal (NOT ...AugLabValSynth).
# See atlas_liver_hcc/trainers/auglab_dualval.py.
#
# checkpoint_final.pth is identical between the val000/val100 mirrors, so a
# final-checkpoint eval only needs ONE of 05_07/05_08 — this wrapper matters for
# checkpoint_best comparisons.
#
# Usage:
#   bash 05_08_predict_auglabAug_v26_6_2_train050_val100_dualval.sh <RUN_ID with _val100_> [FOLD]
# Example:
#   bash 05_08_predict_auglabAug_v26_6_2_train050_val100_dualval.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_<TS> all
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDualVal"
DATASET_ID="80"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
