#!/usr/bin/env bash
# Evaluate one chaos-model run on AMOS CT+MRI: Dice + HD95 for 4 organs.
#
# IMPORTANT — cross-dataset evaluation:
#   Predictions are from chaos-trained models (chaos label IDs).
#   AMOS GT uses different label IDs for the same organs.
#   06_00_evaluate_amos.py handles the remapping explicitly — see its header.
#   Scoreable organs: liver, right_kidney, left_kidney, spleen.
#
# Predictions: PREDICTIONS_ROOT/chaos_models/{CATEGORY}/{RUN_ID}/fold{k}/{mod}/
# GT:          2_nnUNet_amos/raw/labelsTs_{mod}/
# Metrics:     METRICS_ROOT/chaos_models_{CATEGORY}_{RUN_ID}/fold{k}/{mod}_metrics.csv
#                                                                        eval_all.csv
#                                                                        eval_summary.md
#
# Usage:
#   bash 06_01_evaluate_run.sh <CATEGORY> <RUN_ID> [FOLD] [MODALITIES...]
#   FOLD: 0-3 or "all" (default: all)
#   MODALITIES: ct mri or subset (default: ct mri)
#
# Examples:
#   bash 06_01_evaluate_run.sh nnUNet chaos_v26_6_2_train090_val000_20260614_205937
#   bash 06_01_evaluate_run.sh auglab chaos_synthseg_EM_train100_val000_20260611_120000 all ct
#
# Evaluation is CPU-only (no GPU needed) — each modality is launched through
# run_job() (scripts/job_runner/run_job.sh, sourced transitively via
# 00_utils/env.sh) with --gpus 0 --wait, since the per-fold summary script
# below needs every modality's CSV to actually exist before it runs.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

CATEGORY="${1:?CATEGORY required (nnUNet|auglab)}"
RUN_ID="${2:?RUN_ID required}"
FOLD="${3:-all}"
shift $(( $# >= 3 ? 3 : $# )) || true
MODALITIES=("$@"); [ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(ct mri)

EVALUATE_PY="${PROJECT_ROOT}/datasets/amos/5_scripts_amos/06_evaluate/06_00_evaluate_amos.py"
PRED_BASE="${PREDICTIONS_ROOT}/chaos_models/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/chaos_models_${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F} | modalities: ${MODALITIES[*]}"

    local pids=() mod_ok=()
    for mod in "${MODALITIES[@]}"; do
        local PRED_DIR="${PRED_BASE}/fold${F}/${mod}"
        local GT_DIR="${nnUNet_raw}/labelsTs_${mod}"
        if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
            echo "  ! fold${F} ${mod}: no predictions at $PRED_DIR — skipping" >&2
            continue
        fi
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${mod}: no GT dir $GT_DIR — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mod_ok+=("$mod")
        run_job --name "amos_eval_${RUN_ID}_fold${F}_${mod}" --gpus 0 --slot "${SLOT}" --wait -- \
            .venv/bin/python "$EVALUATE_PY" \
            --pred_dir "$PRED_DIR" \
            --gt_dir   "$GT_DIR" \
            --name     "$mod" \
            --out_csv  "${EVAL_DIR}/${mod}_metrics.csv" \
            --workers  8 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    [ ${#mod_ok[@]} -eq 0 ] && { echo "  ! fold${F}: nothing to evaluate"; return; }

    # Merge per-modality CSVs → eval_all.csv + per-fold summary
    .venv/bin/python - "$EVAL_DIR" "$RUN_ID" "$F" "${mod_ok[@]}" <<'PY'
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
    w = csv.DictWriter(f, fieldnames=["group","case","label","dice","hd95"])
    w.writeheader(); w.writerows(rows)

organs = sorted({r["label"] for r in rows})
lines = [f"# Evaluation — {run_id} fold {fold} | AMOS CT+MRI | chaos-trained models",
         "",
         "Predictions from chaos-trained models (chaos label IDs remapped → AMOS IDs).",
         f"Modalities: {', '.join(mods)} | Organs: {', '.join(organs)}", ""]
for metric, title in (("dice","Dice (↑)"),("hd95","HD95 mm (↓)")):
    lines += [f"## {title}","",
              "| modality | " + " | ".join(organs) + " |",
              "|" + "---|"*(len(organs)+1)]
    for m in mods:
        cells = []
        for org in organs:
            v = np.array([float(r[metric]) for r in rows
                          if r["group"]==m and r["label"]==org], float)
            n = int(np.isfinite(v).sum())
            cells.append("—" if not n else
                         f"{np.nanmean(v):.4f}±{np.nanstd(v):.4f}" if metric=="dice"
                         else f"{np.nanmean(v):.2f}±{np.nanstd(v):.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
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
