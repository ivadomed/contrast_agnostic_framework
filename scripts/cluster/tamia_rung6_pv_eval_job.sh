#!/bin/bash
# CPU EVAL job body for ONE rung-6 (PALETTE + boundary PV) run, behind its pack's predict job
# (`tamia_pack_rung6_pv.sh queue-post`). ROW (the run-table name) + OUT exported. The run goes through ITS OWN
# dataset's standard evaluate script with the same settings its ladder's rung 5 used (no hand-rolled eval);
# RUN_JOB_INLINE=1 makes those scripts' own run_job calls execute inside this job. Ladder rungs go to
# ablations/ where the script supports it (on-harmony's testset eval has no METRICS_SUBDIR: flat, like its
# rung 5). Then AUDITED against the rung-5 runs' scored-case counts. Run table: tamia_rung6_pv_runs.sh.
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
set -uo pipefail
: "${ROW:?}" "${OUT:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
source "${ROSTER:-scripts/cluster/tamia_rung6_pv_runs.sh}"   # run table (default: rung 6)
T=benchmark/02_tasks
export RUN_JOB_INLINE=1
row=""; for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack f3 f4 f5 f6 f7 f8 run rest <<<"$r"
  [ "$name" = "$ROW" ] && row="${name}|${pack}|${f3}|${f4}|${f5}|${f6}|${f7}|${f8}|$(rung_resolve_run "${pack}" "${run}")|${rest}"; done
[ -n "$row" ] || { echo "[eval-job] ERROR: no row ${ROW}" >&2; exit 2; }
IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$row"
S="$T/$dir/06_evaluate"
echo "[eval-job] ${name} run=${run} kind=${ekind} host=$(hostname) job=${SLURM_JOB_ID:-?}"

envsetup() { export TRAINING_CONTRAST="$tc"; source "$T/$dir/00_utils/$envf"; source "scripts/cluster/$tenv"; export TRAINING_CONTRAST="$tc"; }

audit() {   # audit <metrics dir to search> <run_id> "item:n ..."  (exactly one nnUNet_<run>/fold{k}/eval_all.csv per fold)
.venv/bin/python - "$1" "$2" "${RUNG_CAT}" $3 <<'PY'
import csv, glob, sys, collections
root, run, cat, exp = sys.argv[1], sys.argv[2], sys.argv[3], dict(x.rsplit(":", 1) for x in sys.argv[4:])
bad = 0
for k in (0, 1, 2):
    ps = glob.glob(f"{root}/**/{cat}_{run}/fold{k}/eval_all.csv", recursive=True)
    if len(ps) != 1:
        print(f"AUDIT FAIL fold{k}: {len(ps)} eval_all.csv found {ps}"); bad = 1; continue
    print(f"fold{k}: {ps[0]}")
    cases = collections.defaultdict(set)
    for r in csv.DictReader(open(ps[0])):
        cases[r.get("group") or r.get("contrast")].add(r["case"])
    for item, n in exp.items():
        if len(cases.get(item, ())) != int(n):
            print(f"AUDIT FAIL fold{k} {item}: {len(cases.get(item, ()))}/{n} cases"); bad = 1
print("AUDIT OK" if not bad else "AUDIT FAILED"); sys.exit(bad)
PY
}

rc=0
case "${ekind}" in
  brats:*)   # in-process, folds sequential (3 folds x 4 contrasts at once OOM-killed a 64G job before)
    rung6_eval_pre_env "${ekind}"; envsetup; export DATASET_ID="${ekind#brats:}" EVAL_INLINE=1 CATEGORY="${RUNG_CAT}" METRICS_SUBDIR="${RUNG_MSUB}"
    for F in 0 1 2; do bash "$S/06_01_evaluate_run.sh" "${run}" "${F}" || rc=1; done ;;
  onh)       # WORKER mode, one fold at a time (default CHECKPOINT=checkpoint_final, as for its rung 5)
    envsetup
    for F in 0 1 2; do bash "$S/06_01_evaluate_testset.sh" "${run}" "${F}" || rc=1; done ;;
  chaos:*)   # DATASET_ID explicit: the non-primary contrast scores against the wrong GT otherwise
    envsetup; export DATASET_ID="${ekind#chaos:}" CATEGORY="${RUNG_CAT}" METRICS_SUBDIR="${RUNG_MSUB}"
    bash "$S/06_01_evaluate_run.sh" "${run}" || rc=1 ;;
  openms_flair)
    envsetup; export METRICS_SUBDIR="${RUNG_MSUB}"
    bash "$S/06_01_evaluate_run.sh" "${run}" "${RUNG_CAT}" || rc=1 ;;
  openms_t1w)
    envsetup; export METRICS_SUBDIR="${RUNG_MSUB}"
    bash "$S/06_13_evaluate_t1w.sh" "${run}" "${RUNG_CAT}" || rc=1 ;;
  ispy2:*)
    envsetup; export METRICS_SUBDIR="${RUNG_MSUB}"
    bash "$S/06_01_evaluate_own_run.sh" "${run}" "${RUNG_CAT}" "${ekind#ispy2:}" all || rc=1 ;;
  pelvic)
    envsetup; export METRICS_SUBDIR="${RUNG_MSUB}"
    bash "$S/06_01_evaluate_run.sh" "${run}" "${RUNG_CAT}" || rc=1 ;;
  tf2)       # mandible-only in-domain; mirrors the run's 3-class metrics location -> seed it under ablations/
    [ -z "${RUNG_MSUB}" ] || mkdir -p "${SCRATCH}/toothfairy2/8_results/02_metrics/toothfairy2_model/cbct/${RUNG_MSUB}/${RUNG_CAT}_${run}"
    ONLY_RUN="${run}" bash "$S/06_08_evaluate_cbct.sh" || rc=1 ;;
  hanseg)    # flipped arm: same settings as its rung 5 (fold{k}/sif/<item>, sif GT, mandible_only_sif tree)
    [ -z "${RUNG_MSUB}" ] || mkdir -p "${SCRATCH}/hanseg/8_results/02_metrics/toothfairy2_model/cbct/${RUNG_MSUB}/${RUNG_CAT}_${run}"
    ONLY_RUN="${run}" PRED_SUBDIR=sif GT_OVERRIDE="${SCRATCH}/hanseg/2_nnUNet/raw/labelsTs_ct_sif" MO_DIR=mandible_only_sif \
      bash "$S/06_03_eval_mandible_only.sh" || rc=1 ;;
  pddca)
    ONLY_RUN="${run}" PRED_SUBDIR=sif GT_OVERRIDE="${SCRATCH}/pddca/2_nnUNet/raw/labelsTs_ct_sif" MO_DIR=mandible_only_sif \
      bash "$S/06_01_eval_mandible_only.sh" || rc=1 ;;
  *) echo "[eval-job] unknown eval kind ${ekind}" >&2; exit 2 ;;
esac

case "${ekind}" in
  tf2)    MROOT="${SCRATCH}/toothfairy2/8_results/02_metrics/toothfairy2_model/cbct_mandible_only" ;;
  hanseg) MROOT="${SCRATCH}/hanseg/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif" ;;
  pddca)  MROOT="${SCRATCH}/pddca/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif" ;;
  *)      MROOT="${METRICS_ROOT}/${MODEL_TYPE}/${tc}" ;;
esac
audit "${MROOT}" "${run}" "${eexp}"; arc=$?
echo "[eval-job] done eval_rc=${rc} audit_rc=${arc}"
[ "${rc}" = 0 ] && [ "${arc}" = 0 ]
