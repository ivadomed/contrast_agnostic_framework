#!/usr/bin/env bash
# Evaluate one chaos-model run on SLIVER07 CT: Dice + HD95 for the liver label only.
#
# SLIVER07 GT annotates the liver alone (label 1). Chaos models emit 4 labels but
# we restrict scoring to liver via --labels liver on the chaos dataset.json.
# The chaos evaluate.py is method-agnostic and reused directly (no copy needed).
#
# Predictions are expected under:
#   PREDICTIONS_ROOT/chaos_models/{CATEGORY}/{RUN_ID}/fold{k}/ct/
# GT is under:
#   2_nnUNet_sliver07/raw/labelsTs_ct/
# Metrics written to:
#   METRICS_ROOT/chaos_models_{CATEGORY}_{RUN_ID}/fold{k}/ct_metrics.csv
#                                                          eval_all.csv
#                                                          eval_summary.md
#
# Usage:
#   bash 06_01_evaluate_run.sh <CATEGORY> <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default: all, 4 folds parallel)
#
# Examples:
#   bash 06_01_evaluate_run.sh nnUNet chaos_t1in_baseline_20260614_153230
#   bash 06_01_evaluate_run.sh auglab chaos_t1in_synthseg_EM_train100_val000_20260611_120000 all
#   bash 06_01_evaluate_run.sh nnUNet chaos_t1in_v26_6_2_train090_val000_20260614_205937 2
#
# Evaluation is CPU-only (no GPU needed) — launched through run_job()
# (scripts/job_runner/run_job.sh, sourced transitively via 00_utils/env.sh)
# with --gpus 0 --wait, since the per-fold summary right after needs the CSV
# to already exist.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

CATEGORY="${1:?CATEGORY required (nnUNet|auglab)}"
RUN_ID="${2:?RUN_ID required (chaos training run dir name)}"
FOLD="${3:-all}"

EVALUATE_PY="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/06_00_evaluate.py"
GT_DIR="${nnUNet_raw}/labelsTs_ct"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -d "$GT_DIR" ]    || { echo "ERROR: GT dir missing: $GT_DIR — run 05_00_build_test_inputs.py" >&2; exit 1; }
[ -f "$EVALUATE_PY" ] || { echo "ERROR: evaluate script not found: $EVALUATE_PY" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_DIR="${PRED_BASE}/fold${F}/ct"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"

    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
        echo "  ! fold${F}: no predictions at $PRED_DIR — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F}"

    run_job --name "sliver07_eval_${RUN_ID}_fold${F}" --gpus 0 --slot "${SLOT}" --wait -- \
        .venv/bin/python "$EVALUATE_PY" \
        --pred_dir  "$PRED_DIR" \
        --gt_dir    "$GT_DIR" \
        --dataset_json "$CHAOS_DATASET_JSON" \
        --labels liver \
        --name ct \
        --out_csv "${EVAL_DIR}/ct_metrics.csv" \
        --workers 8

    # Alias to eval_all.csv so 06_03_aggregate_results.py can find it.
    cp "${EVAL_DIR}/ct_metrics.csv" "${EVAL_DIR}/eval_all.csv"

    # Per-fold summary.
    .venv/bin/python - "${EVAL_DIR}" "${RUN_ID}" "${F}" <<'PY'
import csv, sys
from pathlib import Path
import numpy as np

eval_dir, run_id, fold = sys.argv[1], sys.argv[2], sys.argv[3]
eval_dir = Path(eval_dir)
rows = []
with (eval_dir / "eval_all.csv").open() as f:
    rows = list(csv.DictReader(f))

labels = sorted({r["label"] for r in rows})
lines = [f"# Evaluation — {run_id} fold {fold} | SLIVER07 CT | liver only", "",
         f"Cases: {len({r['case'] for r in rows})} | Label: liver (binarised from chaos 4-label output)", ""]
for metric, title in (("dice", "Dice (↑)"), ("hd95", "HD95 mm (↓)")):
    vals = np.array([float(r[metric]) for r in rows], float)
    n = int(np.isfinite(vals).sum())
    mn, sd = np.nanmean(vals), np.nanstd(vals)
    lines.append(f"**{title}**: {mn:.4f}±{sd:.4f}  (n={n})")
lines.append("")

out = eval_dir / "eval_summary.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"→ {out}")
PY
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} | ALL FOLDS (parallel)"
    for F in 0 1 2 3; do eval_fold "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${METRICS_BASE}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
