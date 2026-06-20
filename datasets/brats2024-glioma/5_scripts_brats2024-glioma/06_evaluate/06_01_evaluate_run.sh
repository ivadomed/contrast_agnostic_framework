#!/usr/bin/env bash
# Evaluate a prediction run against ground truth: Dice + HD95 per contrast & label.
#
# Reads predictions from:  PREDICTIONS_ROOT/CATEGORY/RUN_ID/fold{k}/{contrast}/
# Writes metrics to:       METRICS_ROOT/{CATEGORY}_{RUN_ID}/fold{k}/
#
# Usage:
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD]
#   FOLD can be a number 0-3 or "all" (default: all).
#
# Optional env override:
#   CATEGORY   "nnUNet" (default) or "auglab"
#
# Examples:
#   bash 06_01_evaluate_run.sh brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445              # nnUNet, all folds
#   bash 06_01_evaluate_run.sh brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445 2            # fold 2 only
#   CATEGORY=auglab bash 06_01_evaluate_run.sh brats2024-glioma_t1n_auglab_default_...   # auglab run
#
# Per fold, writes to METRICS_ROOT/{CATEGORY}_{RUN_ID}/fold{k}/:
#   <contrast>_metrics.csv   per-case, per-label Dice & HD95
#   eval_all.csv             all contrasts concatenated
#   eval_summary.md          contrast × label table of mean±std Dice / HD95
#
# Evaluation is CPU-only (no GPU needed) — launched through run_job()
# (scripts/job_runner/run_job.sh, sourced transitively via 00_utils/env.sh)
# with --gpus 0 --wait, since the per-fold summary right after needs the CSV
# to already exist.

set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required}"
FOLD="${2:-all}"
DATASET_ID="${DATASET_ID:-051}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# CATEGORY: use the env override if given, else auto-detect by finding which
# PREDICTIONS_ROOT/{MODEL_TYPE}/{TRAINING_CONTRAST}/<category>/ subdir contains this RUN_ID.
if [ -z "${CATEGORY:-}" ]; then
    _matches=()
    for _c in "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"/*/; do
        [ -d "${_c}${RUN_ID}" ] && _matches+=("$(basename "$_c")")
    done
    case "${#_matches[@]}" in
        1) CATEGORY="${_matches[0]}";;
        0) echo "ERROR: RUN_ID '${RUN_ID}' not found under any ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/<category>/" >&2; exit 1;;
        *) echo "ERROR: RUN_ID '${RUN_ID}' found in multiple categories: ${_matches[*]}. Set CATEGORY=<one> explicitly." >&2; exit 1;;
    esac
    echo "[$(date '+%H:%M:%S')] auto-detected CATEGORY=${CATEGORY} for ${RUN_ID}"
fi

_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
GT_DIR="${nnUNet_raw}/${_DS_NAME}/labelsTr"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="${2:-0}"
    local PRED_ROOT="${PRED_BASE}/fold${F}"
    local EVAL_DIR="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}/fold${F}"

    if [ ! -d "$PRED_ROOT" ]; then
        echo "  ! fold${F}: no predictions dir at $PRED_ROOT — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${RUN_ID} fold${F}  (GT: ${GT_DIR})"

    local contrasts=() pids=()
    for d in "$PRED_ROOT"/*/; do
        local c; c="$(basename "$d")"
        [ "$c" = "eval" ] && continue
        [ -n "$(ls -A "$d"/*.nii.gz 2>/dev/null)" ] || continue
        contrasts+=("$c")
        run_job --name "brats_eval_${RUN_ID}_fold${F}_${c}" \
            --gpus 0 --slot "${SLOT}" --wait -- \
            .venv/bin/python "${HERE}/06_00_evaluate.py" \
            --pred_dir "$d" --gt_dir "$GT_DIR" --dataset_json "$DJ" \
            --name "$c" --out_csv "${EVAL_DIR}/${c}_metrics.csv" \
            --workers 16 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

    if [ ${#contrasts[@]} -eq 0 ]; then
        echo "  ! fold${F}: no contrast predictions found — skipping summary" >&2
        return
    fi

    # per-fold summary: aggregate per-contrast CSVs → combined CSV + markdown table
    .venv/bin/python - "$EVAL_DIR" "$RUN_ID" "$F" "${contrasts[@]}" << 'PY'
import csv, sys
from pathlib import Path
import numpy as np

eval_dir, run_id, fold, *contrasts = sys.argv[1:]
eval_dir = Path(eval_dir)

rows = []
for c in contrasts:
    p = eval_dir / f"{c}_metrics.csv"
    if p.exists():
        with p.open() as f:
            rows.extend(list(csv.DictReader(f)))

with (eval_dir / "eval_all.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
    w.writeheader(); w.writerows(rows)

labels = sorted({r["label"] for r in rows})
def agg(c, lab, key):
    v = np.array([float(r[key]) for r in rows
                  if r["group"] == c and r["label"] == lab], float)
    n = int(np.isfinite(v).sum())
    return (np.nanmean(v) if n else float("nan"),
            np.nanstd(v) if n else float("nan"), n)

lines = [f"# Evaluation — {run_id} (fold {fold})", "",
         f"Contrasts: {', '.join(contrasts)}  |  Labels: {', '.join(labels)}", ""]
for metric, fmt in (("dice", "{:.4f}±{:.4f}"), ("hd95", "{:.2f}±{:.2f}")):
    title = "Dice (↑)" if metric == "dice" else "HD95 mm (↓)"
    lines += [f"## {title}", "",
              "| contrast | " + " | ".join(labels) + " |",
              "|" + "---|" * (len(labels) + 1)]
    for c in contrasts:
        cells = []
        for lab in labels:
            m, s, n = agg(c, lab, metric)
            cells.append("—" if not np.isfinite(m) else fmt.format(m, s))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")

(eval_dir / "eval_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
print(f"→ {eval_dir}/eval_summary.md")
PY
    echo "[$(date '+%H:%M:%S')] fold${F} evaluation done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} | ALL FOLDS (parallel, fold→slot)"
    for F in 0 1 2 3; do
        eval_fold "$F" "$F" &
    done
    wait
    echo "[$(date '+%H:%M:%S')] all folds evaluated → ${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
