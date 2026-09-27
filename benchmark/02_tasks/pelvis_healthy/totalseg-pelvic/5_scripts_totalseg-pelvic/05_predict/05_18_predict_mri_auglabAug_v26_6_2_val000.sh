#!/usr/bin/env bash
# Predict with the MRI-trained auglabAug_v26_6_2_val000 model, over ct/mri.
# Usage: bash 05_18_predict_mri_auglabAug_v26_6_2_val000.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_mri.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="130"   # fixed — imagesTs_* consolidated under Dataset130, see 05_01 header
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
