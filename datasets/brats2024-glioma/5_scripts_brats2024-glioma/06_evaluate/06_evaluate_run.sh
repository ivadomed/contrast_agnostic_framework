#!/usr/bin/env bash
# Evaluate a prediction run against ground truth: Dice + HD95 per contrast & label.
#
# Method-agnostic — it just walks predictions/fold<F>/<contrast>/ and compares each
# against the shared ground-truth labels. Works for any experiment/run.
#
# Usage:
#   bash 06_evaluate_run.sh <RUN_ID> [FOLD]
# Example:
#   bash 06_evaluate_run.sh v26_6_2_20260608_003445 0
#
# Writes, under predictions/fold<F>/eval/ :
#   <contrast>_metrics.csv   per-case, per-label Dice & HD95
#   eval_all.csv             all contrasts concatenated
#   eval_summary.md          contrast × label table of mean±std Dice / HD95

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/../00_utils/env.sh"

RUN_ID="${1:?RUN_ID required}"
FOLD="${2:-0}"
DATASET_ID="${DATASET_ID:-051}"
SLOT="${SLOT:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"

_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
GT_DIR="${nnUNet_raw}/${_DS_NAME}/labelsTr"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_ROOT="${nnUNet_results}/${RUN_ID}/predictions/fold${FOLD}"
EVAL_DIR="${PRED_ROOT}/eval"

[ -d "$PRED_ROOT" ] || { echo "ERROR: no predictions at $PRED_ROOT" >&2; exit 1; }
mkdir -p "$EVAL_DIR"

echo "[$(date '+%H:%M:%S')] evaluate ${RUN_ID} fold${FOLD}  (GT: ${GT_DIR})"
contrasts=()
for d in "$PRED_ROOT"/*/; do
    c="$(basename "$d")"
    [ "$c" = "eval" ] && continue
    [ -n "$(ls -A "$d"/*.nii.gz 2>/dev/null)" ] || continue
    contrasts+=("$c")
    set_slot ${SLOT} "$(pwd)/.venv/bin/python" "${HERE}/06_evaluate.py" \
        --pred_dir "$d" --gt_dir "$GT_DIR" --dataset_json "$DJ" \
        --name "$c" --out_csv "${EVAL_DIR}/${c}_metrics.csv"
done

# aggregate per-contrast CSVs → combined CSV + markdown summary table
"$(pwd)/.venv/bin/python" - "$EVAL_DIR" "$RUN_ID" "$FOLD" "${contrasts[@]}" << 'PY'
import csv, sys
from pathlib import Path
import numpy as np

eval_dir, run_id, fold, *contrasts = sys.argv[1:]
eval_dir = Path(eval_dir)

rows = []
for c in contrasts:
    with (eval_dir / f"{c}_metrics.csv").open() as f:
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

echo "[$(date '+%H:%M:%S')] evaluation done → ${EVAL_DIR}/"
