#!/usr/bin/env bash
# Evaluate one pansegdata prediction run (all folds x test item(s)), scoring against
# nnUNet_raw ground truth. Writes eval_all.csv + eval_summary.md per fold under
#   8_results_pansegdata/02_metrics/pansegdata_model/<contrast>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# TODO this is a scaffolded stub, not a working script yet -- fill in:
#   - ITEMS (test item names, e.g. one per contrast/modality)
#   - LABELS (this dataset's foreground label names, matching dataset.json)
#   - DATASET_ID / GT_DIR (where nnUNet_raw ground truth actually lives)
# See totalseg-pelvic's or toothfairy2's 06_01_evaluate_run.sh for a complete
# reference implementation of this same pattern.
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
#
# CATEGORY must be passed explicitly and correctly -- a wrong CATEGORY does NOT error,
# it silently writes an empty, _logs-only metrics dir. Check eval_all.csv actually
# exists before trusting a run finished.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

echo "TODO: 06_01_evaluate_run.sh for pansegdata is a scaffolded stub -- fill in ITEMS/LABELS/GT_DIR" >&2
exit 1
