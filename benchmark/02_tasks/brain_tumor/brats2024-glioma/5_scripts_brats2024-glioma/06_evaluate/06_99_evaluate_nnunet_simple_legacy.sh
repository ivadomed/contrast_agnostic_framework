#!/usr/bin/env bash
# Evaluate predictions against ground-truth on the 70 held-out test cases.
# Expected env vars: METHOD, RUN_ID, DATASET_ID

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

DATASET_ID="${DATASET_ID:-051}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
GT_DIR="${nnUNet_raw}/${_DS_NAME}/labelsTr"
PRED_DIR="${nnUNet_results}/${RUN_ID}/predictions/${_DS_NAME}"
EVAL_OUT="${nnUNet_results}/${RUN_ID}/predictions/${_DS_NAME}/eval_summary.json"

echo "[$(date '+%H:%M:%S')] Evaluating ${METHOD} (${RUN_ID})"
echo "  Predictions: $PRED_DIR"
echo "  Ground truth: $GT_DIR"

.venv/bin/nnUNetv2_evaluate_simple \
    -gt "$GT_DIR" \
    -p "$PRED_DIR" \
    -djfile "${nnUNet_raw}/${_DS_NAME}/dataset.json" \
    -pfile "${nnUNet_preprocessed}/${_DS_NAME}/nnUNetPlans.json" \
    -o "$EVAL_OUT"

echo "[$(date '+%H:%M:%S')] Evaluation done → $EVAL_OUT"
python3 - "$EVAL_OUT" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
mean = d.get("mean", {})
print("\nDice per class (mean over test set):")
for cls, vals in mean.items():
    if isinstance(vals, dict):
        print(f"  {cls}: {vals.get('Dice', 'N/A'):.4f}")
PYEOF
