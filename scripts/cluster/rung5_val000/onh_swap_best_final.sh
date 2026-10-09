#!/bin/bash
# on-harmony checkpoint_final -> checkpoint_best switch (2026-10-06), STEP C: the metrics-dir swap, run ONCE after
# onh_best_predict_eval.sh verified every roster run. Afterwards on-harmony follows the shared convention:
#   <category>_<run>        = checkpoint_best (no TTA, like every dataset)
#   <category>_<run>_final  = the legacy checkpoint_final metrics (predicted WITH mirroring TTA by the pre-2026-09-27
#                             inline script; kept for the record, read by the best_vs_final checkpoint-comparison configs)
# Pass 1: every unsuffixed on-harmony run metrics dir -> <dir>_final (ALL of them, also runs without a best arm, so an
#         unsuffixed dir always means best). Pass 2: <dir>_best -> <dir>. checkpoint_comparison/ (the 07-31 sweep) and
#         toDelete/ are not touched. Refuses to overwrite anything. Metrics only (predictions were moved by 05_90).
#   bash scripts/cluster/rung5_val000/onh_swap_best_final.sh [--dry-run]
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
M=benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
mvv() { [ -e "$2" ] && ! { [ "$DRY" = 1 ] && [ -e "$2_final" -o -d "$2" ] && [ "${PASS}" = 2 ]; } && { echo "REFUSE: $2 exists" >&2; exit 1; }; if [ "$DRY" = 1 ]; then echo "would mv $1 -> $2"; else mv "$1" "$2"; fi; }
dirs() { for d in "$M"/{T1w,T2w,dwi_ap}/{nnUNet,auglab}_on-harmony_* "$M"/{T1w,T2w,dwi_ap}/ablations/{nnUNet,auglab}_on-harmony_*; do
           [ -d "$d" ] || continue; case "$d" in *.bak*|*.stale_*) continue ;; esac; echo "$d"; done; }
n1=0; n2=0; PASS=1
for d in $(dirs); do case "$d" in *_best|*_final) ;; *) mvv "$d" "${d}_final"; n1=$((n1+1)) ;; esac; done
PASS=2
for d in $(dirs); do case "$d" in *_best) mvv "$d" "${d%_best}"; n2=$((n2+1)) ;; esac; done
echo "[swap] pass 1 (-> _final): ${n1}; pass 2 (_best -> unsuffixed): ${n2} $([ "$DRY" = 1 ] && echo '(dry run)')"
