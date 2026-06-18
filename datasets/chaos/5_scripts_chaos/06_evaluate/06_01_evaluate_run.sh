#!/usr/bin/env bash
# Evaluate a prediction run against ground truth: Dice + HD95 per modality & label.
# Ported from brats2024-glioma/06_evaluate_run.sh, adapted for CHAOS's PER-MODALITY GT
# (each modality has its own labelsTs_<mod> — unlike BraTS's single labelsTr).
#
# Reads predictions from:  PREDICTIONS_ROOT/CATEGORY/RUN_ID/fold{k}/{modality}/
# Ground truth per modality: nnUNet_raw/<DS>/labelsTs_{modality}/
# Writes metrics to:        METRICS_ROOT/{CATEGORY}_{RUN_ID}/fold{k}/
#
# Usage:
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default: all).
# Optional env override:
#   CATEGORY   "nnUNet" or "auglab" (default: auto-detected from PREDICTIONS_ROOT)
#
# Examples:
#   bash 06_01_evaluate_run.sh chaos_v26_6_2_train090_val000_20260614_205937   # all folds
#   bash 06_01_evaluate_run.sh chaos_auglab_default_20260611_120000 2          # fold 2 only
#   CATEGORY=auglab bash 06_01_evaluate_run.sh chaos_synthseg_EM_train100_val000_20260611_120000
#
# Per fold → METRICS_ROOT/{CATEGORY}_{RUN_ID}/fold{k}/:
#   <modality>_metrics.csv   per-case, per-label Dice & HD95
#   eval_all.csv             all modalities concatenated (consumed by 06_02_aggregate)
#   eval_summary.md          modality × label table of mean±std Dice / HD95
#
# NOTE: CT GT is liver-only — only the `liver` rows are meaningful for the ct modality;
# kidney/spleen rows there are spurious (read liver-only for CT).

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/../00_utils/env.sh"

RUN_ID="${1:?RUN_ID required}"
FOLD="${2:-all}"
DATASET_ID="${DATASET_ID:-60}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# CATEGORY: env override if given, else auto-detect which PREDICTIONS_ROOT/<category>/
# subdir contains this RUN_ID.
if [ -z "${CATEGORY:-}" ]; then
    _matches=()
    for _c in "${PREDICTIONS_ROOT}"/*/; do
        [ -d "${_c}${RUN_ID}" ] && _matches+=("$(basename "$_c")")
    done
    case "${#_matches[@]}" in
        1) CATEGORY="${_matches[0]}";;
        0) echo "ERROR: RUN_ID '${RUN_ID}' not found under any ${PREDICTIONS_ROOT}/<category>/" >&2; exit 1;;
        *) echo "ERROR: RUN_ID '${RUN_ID}' in multiple categories: ${_matches[*]}. Set CATEGORY=<one>." >&2; exit 1;;
    esac
    echo "[$(date '+%H:%M:%S')] auto-detected CATEGORY=${CATEGORY} for ${RUN_ID}"
fi

_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${CATEGORY}/${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

# Scoreable label names per modality, from test_cases.json's scoreable_organs:
# CT GT annotates only the liver → score liver-only for ct (kidneys/spleen are NOT
# present and must be excluded, not counted as 0.0). MR modalities score all organs.
declare -A MOD_LABELS
while IFS=$'\t' read -r _mod _labs; do
    [ -n "$_mod" ] && MOD_LABELS["$_mod"]="$_labs"
done < <("$(pwd)/.venv/bin/python" - "$DJ" "${SPLITS_DIR}/test_cases.json" <<'PY'
import json, sys
dj = json.loads(open(sys.argv[1]).read())
tc = json.loads(open(sys.argv[2]).read())
id2name = {int(v): k for k, v in dj["labels"].items() if isinstance(v, int) and int(v) != 0}
sc = tc.get("scoreable_organs", {})
for src, mods_key in (("MR", "mr_test_modalities"), ("CT", "ct_test_modalities")):
    names = " ".join(id2name[i] for i in sc.get(src, []) if i in id2name)
    for mod in tc.get(mods_key, []):
        print(f"{mod}\t{names}")
PY
)

eval_fold() {
    local F="$1" SLOT="${2:-0}"
    local PRED_ROOT="${PRED_BASE}/fold${F}"
    local EVAL_DIR="${METRICS_ROOT}/${CATEGORY}_${RUN_ID}/fold${F}"

    if [ ! -d "$PRED_ROOT" ]; then
        echo "  ! fold${F}: no predictions dir at $PRED_ROOT — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${RUN_ID} fold${F}"

    local mods=() pids=()
    for d in "$PRED_ROOT"/*/; do
        local m; m="$(basename "$d")"
        [ -n "$(ls -A "$d"/*.nii.gz 2>/dev/null)" ] || continue
        local GT_DIR="${nnUNet_raw}/${_DS_NAME}/labelsTs_${m}"   # per-modality GT
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${m}: no GT dir ($GT_DIR) — skipping" >&2
            continue
        fi
        mods+=("$m")
        # Restrict to this modality's scoreable labels (ct → liver only).
        local LBL_ARG=""
        [ -n "${MOD_LABELS[$m]:-}" ] && LBL_ARG="--labels ${MOD_LABELS[$m]}"
        set_slot ${SLOT} "$(pwd)/.venv/bin/python" "${HERE}/06_00_evaluate.py" \
            --pred_dir "$d" --gt_dir "$GT_DIR" --dataset_json "$DJ" \
            --name "$m" --out_csv "${EVAL_DIR}/${m}_metrics.csv" \
            ${LBL_ARG} --workers 16 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

    if [ ${#mods[@]} -eq 0 ]; then
        echo "  ! fold${F}: no modality predictions found — skipping summary" >&2
        return
    fi

    # per-fold summary: aggregate per-modality CSVs → combined CSV + markdown table
    "$(pwd)/.venv/bin/python" - "$EVAL_DIR" "$RUN_ID" "$F" "${mods[@]}" << 'PY'
import csv, sys
from pathlib import Path
import numpy as np

eval_dir, run_id, fold, *mods = sys.argv[1:]
eval_dir = Path(eval_dir)

rows = []
for m in mods:
    p = eval_dir / f"{m}_metrics.csv"
    if p.exists():
        with p.open() as f:
            rows.extend(list(csv.DictReader(f)))

with (eval_dir / "eval_all.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
    w.writeheader(); w.writerows(rows)

labels = sorted({r["label"] for r in rows})
def agg(m, lab, key):
    v = np.array([float(r[key]) for r in rows
                  if r["group"] == m and r["label"] == lab], float)
    n = int(np.isfinite(v).sum())
    return (np.nanmean(v) if n else float("nan"),
            np.nanstd(v) if n else float("nan"), n)

lines = [f"# Evaluation — {run_id} (fold {fold})", "",
         f"Modalities: {', '.join(mods)}  |  Labels: {', '.join(labels)}",
         "(CT modality: read `liver` only — kidney/spleen GT absent.)", ""]
for metric, fmt in (("dice", "{:.4f}±{:.4f}"), ("hd95", "{:.2f}±{:.2f}")):
    title = "Dice (↑)" if metric == "dice" else "HD95 mm (↓)"
    lines += [f"## {title}", "",
              "| modality | " + " | ".join(labels) + " |",
              "|" + "---|" * (len(labels) + 1)]
    for m in mods:
        cells = []
        for lab in labels:
            mn, sd, n = agg(m, lab, metric)
            cells.append("—" if not np.isfinite(mn) else fmt.format(mn, sd))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
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
    echo "[$(date '+%H:%M:%S')] all folds evaluated → ${METRICS_ROOT}/${CATEGORY}_${RUN_ID}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
