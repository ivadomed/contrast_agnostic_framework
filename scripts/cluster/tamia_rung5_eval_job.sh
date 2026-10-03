#!/bin/bash
# CPU EVAL job body for the rung-5 retrains (chaos t2spir, on-harmony T1w, brats t2w), run behind the predict job
# (tamia_pack_rung5_retrain.sh queue-post). Each run goes through ITS OWN dataset's standard 06_01 evaluate script
# (no hand-rolled eval), metrics routed to ablations/ via METRICS_SUBDIR, then AUDITED against the old runs' case counts.
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
set -uo pipefail
: "${ROOT:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
source "${ROOT}/RUN_IDS.env"
T=benchmark/02_tasks
export METRICS_SUBDIR=ablations
rc_all=0
audit() {   # audit <metrics_root> <run_id> "item:n ..."
.venv/bin/python - "$1" "$2" $3 <<'PY'
import csv, glob, sys, collections
root, run, exp = sys.argv[1], sys.argv[2], dict(x.split(":") for x in sys.argv[3:])
bad = 0
for k in (0, 1, 2):
    ps = glob.glob(f"{root}/**/ablations/nnUNet_{run}/fold{k}/eval_all.csv", recursive=True)
    if len(ps) != 1:
        print(f"AUDIT FAIL fold{k}: {len(ps)} eval_all.csv found"); bad = 1; continue
    cases = collections.defaultdict(set)
    for r in csv.DictReader(open(ps[0])): cases[r["group"]].add(r["case"])
    for item, n in exp.items():
        if len(cases.get(item, ())) != int(n):
            print(f"AUDIT FAIL fold{k} {item}: {len(cases.get(item, ()))}/{n} cases"); bad = 1
print("AUDIT OK" if not bad else "AUDIT FAILED"); sys.exit(bad)
PY
}

echo "[eval-job] host=$(hostname) job=${SLURM_JOB_ID:-?}"
# ---- chaos t2spir (DATASET_ID explicit: non-primary contrast scores against the wrong GT otherwise) ----
( source "$T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env_t2spir.sh"; source scripts/cluster/tamia_env_chaos.sh
  export TRAINING_CONTRAST=t2spir DATASET_ID=61 CATEGORY=nnUNet
  bash "$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_01_evaluate_run.sh" "${CHAOS_RUN}"
  audit "${METRICS_ROOT}" "${CHAOS_RUN}" "t1in:4 t1out:4 t2spir:4 ct:20" ) || { echo "[eval-job] chaos FAILED"; rc_all=1; }
# ---- on-harmony T1w (WORKER mode, one fold at a time) ----
( source "$T/brain_healthy/on-harmony/5_scripts_on-harmony/00_utils/env.sh"; source scripts/cluster/tamia_env_onharmony.sh
  export TRAINING_CONTRAST=T1w
  rc=0; for F in 0 1 2; do bash "$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_01_evaluate_testset.sh" "${ONH_RUN}" "${F}" || rc=1; done
  audit "${METRICS_ROOT}" "${ONH_RUN}" "T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:8" && [ "$rc" = 0 ] ) || { echo "[eval-job] on-harmony FAILED"; rc_all=1; }
# ---- brats t2w (in-process, folds sequential: 3 folds x 4 contrasts at once OOM-killed a 64G job before) ----
( export METRICS_ROOT="${SCRATCH}/brats2024-glioma/8_results/02_metrics"
  source "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env_t2w.sh"; source scripts/cluster/tamia_env.sh
  export TRAINING_CONTRAST=t2w DATASET_ID=052 EVAL_INLINE=1 CATEGORY=nnUNet
  rc=0; for F in 0 1 2; do bash "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_01_evaluate_run.sh" "${BRATS_RUN}" "${F}" || rc=1; done
  audit "${METRICS_ROOT}" "${BRATS_RUN}" "t1n:70 t1c:70 t2w:70 t2f:70" && [ "$rc" = 0 ] ) || { echo "[eval-job] brats FAILED"; rc_all=1; }
echo "[eval-job] done rc_all=${rc_all}"
exit ${rc_all}
