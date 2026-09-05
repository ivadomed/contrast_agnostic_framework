#!/usr/bin/env bash
# Predict the atlas-liver-hcc DUAL-VAL run's clean/val000 checkpoint (AugLab + V26_6_2
# @50% train). The dual-val run (04_06) produces BOTH val000 and val100 checkpoints in
# ONE training, so BOTH mirror RUN_IDs live under the
# nnUNetTrainerAtlasLiverHCCAugLabDualVal trainer dir. Predict MUST therefore use
# TRAINER=...AugLabDualVal — NOT ...AugLabDefault nor ...AugLabValSynth.
# See atlas_liver_hcc/trainers/auglab_dualval.py.
#
# Usage:
#   bash 05_07_predict_auglabAug_v26_6_2_train050_val000_dualval.sh <RUN_ID with _val000_> [FOLD]
# Example:
#   bash 05_07_predict_auglabAug_v26_6_2_train050_val000_dualval.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_<TS> all
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDualVal"
DATASET_ID="80"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
