#!/bin/bash
# CPU eval job body for ONE brats2024-glioma T2f/FLAIR-trained run (all folds, all contrasts).
# Submitted by 06_20_tamia_eval_t2f.sh with RUN_ID / CATEGORY / METRICS_SUBDIR / HERE_DIR exported.
# Runs the standard 06_01_evaluate_run.sh in-process (EVAL_INLINE=1), then AUDITS the output.
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -uo pipefail
: "${RUN_ID:?}" "${CATEGORY:?}" "${HERE_DIR:?}"
cd "${HERE_DIR}/06_evaluate"
# METRICS_ROOT defaults to the /project path (file-count-quota-limited on TamIA) and tamia_env.sh does not
# override it — set it to scratch BEFORE env.sh is sourced (common_env keeps an already-exported value).
export METRICS_ROOT="/scratch/p/paulh/brats2024-glioma/8_results/02_metrics"
source ../00_utils/env.sh
source ../../../../../../scripts/cluster/tamia_env.sh
source ../00_utils/t2f_runs.sh
export TRAINING_CONTRAST=t2f DATASET_ID=053 EVAL_INLINE=1 CATEGORY METRICS_SUBDIR="${METRICS_SUBDIR:-}"
echo "[eval-job] RUN_ID=${RUN_ID} CATEGORY=${CATEGORY} METRICS_SUBDIR='${METRICS_SUBDIR}' METRICS_ROOT=${METRICS_ROOT} host=$(hostname)"

# Folds run SEQUENTIALLY (4 contrasts in parallel within a fold): "all" runs 3 folds x 4 contrasts x 8 workers at
# once, which OOM-killed a 64G job on the first attempt (2026-09-21, 7 of 12 per-contrast processes killed).
rc=0
for F in 0 1 2; do bash 06_01_evaluate_run.sh "${RUN_ID}" "${F}" || rc=1; done

OUT="${METRICS_ROOT}/${MODEL_TYPE}/t2f${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}"
echo "[eval-job] AUDIT ${OUT}: ${T2F_N_TEST_CASES} unique cases x contrasts [${T2F_CONTRASTS}] per fold 0-2"
"${PROJECT_ROOT}/.venv/bin/python" - "${OUT}" "${T2F_N_TEST_CASES}" ${T2F_CONTRASTS} <<'PY'
import csv, os, sys, collections
out, n_exp, contrasts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
bad = 0
for k in (0, 1, 2):
    p = f"{out}/fold{k}/eval_all.csv"
    if not os.path.exists(p):
        print(f"AUDIT FAIL fold{k}: {p} missing"); bad = 1; continue
    cases = collections.defaultdict(set)
    for r in csv.DictReader(open(p)):
        cases[r["group"]].add(r["case"])
    for c in contrasts:
        n = len(cases.get(c, ()))
        if n != n_exp:
            print(f"AUDIT FAIL fold{k} {c}: {n}/{n_exp} cases"); bad = 1
print("AUDIT OK" if not bad else "AUDIT FAILED")
sys.exit(bad)
PY
audit_rc=$?
echo "[eval-job] done eval_rc=${rc} audit_rc=${audit_rc}"
[ "${rc}" = "0" ] && [ "${audit_rc}" = "0" ]
