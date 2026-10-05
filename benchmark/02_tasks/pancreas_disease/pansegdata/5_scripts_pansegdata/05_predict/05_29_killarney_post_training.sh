#!/usr/bin/env bash
# Post-training CONTROLLER for Killarney (a CPU job queued by 05_30 with --dependency=afterany on all 60 training fold jobs; also runnable by hand).
#  1. verify every roster run (6 headline methods + OURS val100 mirror + 4 ladder rungs, per contrast = 11 x 2) has checkpoint_final.pth for folds 0 1 2;
#     any gap -> write the status file and STOP (exit 2): nothing is predicted on partial training; fix/resume (re-run the wrapper with the SAME RUN_ID), re-run this;
#  2. pin the resolved RUN_IDs into each contrast's roster_run_ids.tsv (predict / eval / configs / ladders all read the pins);
#  3. submit the predict jobs of both contrasts through the shared roster driver (05_24 / 05_25 -> per-wrapper run_job on Killarney GPUs: light inference -> L40S);
#  4. queue the eval stage (06_09) with --dependency=afterany on those predict jobs.
#   bash 05_29_killarney_post_training.sh [--check]       (--check: only report training completeness)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
unset RUN_JOB_DEPENDENCY                       # this job inherits the dependency it was queued with; never pass it on to the jobs it submits
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}" RUN_JOB_GPU_TYPE="${POST_GPU_TYPE:-l40s}" RUN_JOB_TIME_DEFAULT="${POST_PREDICT_TIME:-02:00:00}"
source "${HERE}/../00_utils/env.sh"
OUTD="${RESULTS_DIR}/post_training"; mkdir -p "${OUTD}"
STATUS="${OUTD}/post_training_status.txt"; : > "${STATUS}"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${STATUS}"; }

missing=0
for C in t1wce t2w; do
    ( if [ "${C}" = t2w ]; then source "${HERE}/../00_utils/env_t2w.sh"; else source "${HERE}/../00_utils/env.sh"; fi
      source "${ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
      bad=0; n=0
      for w in "${HERE}"/05_*_predict_${C}_*.sh; do
          case "$w" in *run_all*) continue;; esac
          roster_parse_wrapper "$w" || exit 3
          rid="$(roster_resolve "${W_METHOD}" "${W_CATEGORY}")" || { echo "  [${C}] no trained run for ${W_METHOD} (${W_CATEGORY})"; bad=$((bad+1)); continue; }
          for F in 0 1 2; do
              ls "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${C}/${W_CATEGORY}/${rid}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${F}/checkpoint_final.pth >/dev/null 2>&1 \
                  || { echo "  [${C}] MISSING checkpoint_final: ${W_METHOD} (${rid}) fold ${F}"; bad=$((bad+1)); }
          done
          roster_pin "${W_METHOD}" "${W_CATEGORY}" "${rid}"; n=$((n+1))
      done
      echo "  [${C}] ${n} roster methods resolved and pinned (${bad} problems) -> $(roster_pin_file)"
      exit "${bad}" ) 2>&1 | tee -a "${STATUS}"
    missing=$((missing + ${PIPESTATUS[0]}))
done
if [ "${missing}" != 0 ]; then say "INCOMPLETE TRAINING: ${missing} problems above. Nothing predicted. Fix/resume the missing folds, then re-run this script."; exit 2; fi
say "training complete for both contrasts; pins written"
[ "${1:-}" = "--check" ] && exit 0

ids=""
for C in t1wce t2w; do
    out="$( if [ "${C}" = t2w ]; then source "${HERE}/../00_utils/env_t2w.sh"; else source "${HERE}/../00_utils/env.sh"; fi
            if [ "${C}" = t1wce ]; then bash "${HERE}/05_24_run_all_predict_t1wce.sh" 2>&1; else bash "${HERE}/05_25_run_all_predict_t2w.sh" 2>&1; fi )"; rc=$?
    echo "${out}" | tail -5 | tee -a "${STATUS}"
    [ "${rc}" = 0 ] || { say "ERROR: predict submission for ${C} failed (rc=${rc})"; exit 3; }
    ids="${ids}:$(echo "${out}" | grep -o 'Submitted batch job [0-9]*' | awk '{print $4}' | paste -sd: -)"
done
ids="${ids#:}"; ids="${ids//::/:}"; n_ids=$(echo "${ids}" | tr ':' '\n' | grep -c .)
say "predict jobs submitted: ${n_ids} (${ids})"
[ "${n_ids}" -ge 22 ] || { say "ERROR: expected >= 22 predict jobs (11 runs x 2 contrasts), got ${n_ids}; not queueing eval"; exit 3; }
source "${HERE}/../00_utils/env.sh"
RUN_JOB_DEPENDENCY="afterany:${ids}" RUN_JOB_GPU_TYPE="" run_job --name pansegdata_eval_all --gpus 0 --cpus 8 --mem 32G --time 05:00:00 \
    --log "${OUTD}/eval_stage.log" -- bash "${ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/06_evaluate/06_09_killarney_eval_stage.sh" | tee -a "${STATUS}"
say "eval stage queued (afterany:${n_ids} predict jobs); log ${OUTD}/eval_stage.log"
