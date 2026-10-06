#!/bin/bash
# on-harmony checkpoint_final -> checkpoint_best switch (2026-10-06), STEP B: predict checkpoint_best for every run the
# on-harmony configs / ladders reference, evaluate it, and verify. CPU controller, run through run_job on Vulcan:
#   run_job --name onh_best --gpus 0 --cpus 4 --mem 16G --time 08:00:00 --log $SCRATCH/rung5_val000/onh_fix/onh_best.log -- \
#       bash scripts/cluster/rung5_val000/onh_best_predict_eval.sh <roster.tsv> [run-id-substring filter]
# roster.tsv rows: train_contrast  category  run_id  trainer  n_best_ckpts  metrics_subdir(""|ablations)
# - predictions -> flat fold{k}/<contrast>/ (checkpoint_best layout; the legacy final ones were moved to fold{k}/final/
#   by 05_predict/05_90_migrate_flat_final_predictions.sh). The RAS->native resample runs inside each predict job.
# - metrics -> <category>_<run>_best (the evaluate script's CURRENT suffix for best); STEP C (the swap) renames
#   <run> -> <run>_final and <run>_best -> <run> once every row here is verified.
# - verify: eval_all.csv per fold + per-contrast case counts = the test set (T1w/T2w/bold/dwi_ap 8, epi_ap 4, gre 7).
# Status: $SCRATCH/rung5_val000/onh_fix/status/<run>.status (last line OK / FAILED).
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
ROSTER="${1:?roster.tsv}"; FILTER="${2:-}"
ROOT="$PWD"; S="${ROOT}/benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony"
OUT="${SCRATCH:?}/rung5_val000/onh_fix/status"; mkdir -p "${OUT}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"   # rack02-06: CUDA unknown error, ends COMPLETED w/o output
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE RUN_JOB_PACK_DIR
MAXPAR="${MAXPAR:-8}"

one() {   # one <tc> <cat> <run> <trainer> <nbest> <sub>
  local tc="$1" cat="$2" run="$3" tr="$4" nb="$5" sub="$6" envf st
  st="${OUT}/${run}.status"; : > "${st}"
  log() { echo "[$(date '+%F %T')] $*" >> "${st}"; }
  case "${tc}" in T1w) envf=env.sh ;; T2w) envf=env_t2w.sh ;; dwi_ap) envf=env_dwi.sh ;; esac
  [ "${nb}" = 3 ] || { log "FAILED: ${nb}/3 checkpoint_best"; return 1; }
  log "predict checkpoint_best (${tc} ${cat} ${tr})"
  ( source "${S}/00_utils/${envf}"; export TRAINING_CONTRAST="${tc}" CHECKPOINT=checkpoint_best.pth
    METHOD="${run#on-harmony_${tc}_}"; TRAINER="${tr}"; CATEGORY="${cat}"
    source "${S}/05_predict/05_01_predict_common.sh" "${run}" all ) > "${OUT}/${run}.predict.log" 2>&1 \
    || { log "FAILED: predict (see ${run}.predict.log)"; return 1; }
  local M; M="$(source "${S}/00_utils/${envf}" >/dev/null 2>&1; echo "${METRICS_ROOT}/${MODEL_TYPE}/${tc}${sub:+/${sub}}/${cat}_${run}_best")"
  [ -e "${M}" ] && { mv "${M}" "${M}.stale_$(date +%Y%m%d_%H%M%S)"; log "moved a pre-existing ${M} aside (06_01 would reuse stale CSVs)"; }
  log "evaluate -> ${M}"
  ( source "${S}/00_utils/${envf}"; export CHECKPOINT=checkpoint_best.pth METRICS_SUBDIR="${sub}" RUN_JOB_INLINE=1
    for F in 0 1 2; do bash "${S}/06_evaluate/06_01_evaluate_testset.sh" "${run}" "${F}" || exit 1; done ) > "${OUT}/${run}.eval.log" 2>&1 \
    || { log "FAILED: evaluate (see ${run}.eval.log)"; return 1; }
  "${ROOT}/.venv/bin/python" - "${M}" >> "${st}" 2>&1 <<'PY' || { log "FAILED: verify"; return 1; }
import csv, sys
from pathlib import Path
exp = {"T1w": 8, "T2w": 8, "bold": 8, "dwi_ap": 8, "epi_ap": 4, "gre_echo1_mag": 7}
m = Path(sys.argv[1]); bad = []
for k in range(3):
    f = m / f"fold{k}" / "eval_all.csv"
    if not f.exists():
        bad.append(f"fold{k}: no eval_all.csv"); continue
    cases = {}
    for r in csv.DictReader(f.open()):
        cases.setdefault(r["group"], set()).add(r["case"])
    for c, n in exp.items():
        if len(cases.get(c, ())) != n:
            bad.append(f"fold{k}/{c}: {len(cases.get(c, ()))} cases, expected {n}")
print("verify:", "OK" if not bad else "; ".join(bad)); sys.exit(1 if bad else 0)
PY
  log "OK"
}

n=0
while IFS=$'\t' read -r tc cat run tr nb sub; do
  [ -z "${run}" ] && continue
  [ -n "${FILTER}" ] && [[ "${run}" != *"${FILTER}"* ]] && continue
  one "${tc}" "${cat}" "${run}" "${tr}" "${nb}" "${sub}" &
  n=$((n+1)); while [ "$(jobs -rp | wc -l)" -ge "${MAXPAR}" ]; do wait -n; done
done < "${ROSTER}"
wait
echo "== ${n} runs"; ok=0
while IFS=$'\t' read -r tc cat run rest; do
  [ -n "${FILTER}" ] && [[ "${run}" != *"${FILTER}"* ]] && continue
  l="$(tail -1 "${OUT}/${run}.status" 2>/dev/null)"; echo "${run}: ${l}"; [[ "${l}" == *"] OK" ]] && ok=$((ok+1))
done < "${ROSTER}"
echo "== ${ok}/${n} OK"; [ "${ok}" = "${n}" ]
