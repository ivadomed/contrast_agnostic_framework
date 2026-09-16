#!/usr/bin/env bash
# Predict with the CT-trained auglabAug_v26_6_2_val100 model, over ct/mri.
# Usage: bash 05_08_predict_ct_auglabAug_v26_6_2_val100.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDualVal"
CATEGORY="auglab"
DATASET_ID="130"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
