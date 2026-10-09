#!/usr/bin/env bash
# Post-training CONTROLLER (a CPU job queued by 05_27 to run after every training pack job ends; also runnable by hand on TamIA).
#  1. verify every roster run has checkpoint_final.pth for folds 0 1 2  (incl. the DualVal val100 mirror run) -- any gap: write the status file and
#     STOP (exit 2) instead of predicting on partial training; fix/resume (re-run 04_23), then re-run this script;
#  2. pin the exact RUN_IDs (from $SCRATCH/pansegdata/_packruns/RUN_IDS.tsv) into each contrast's roster_run_ids.tsv, so predict/eval/configs/ladders
#     use the runs that were trained here and never "the newest dir";
#  3. record + submit ONE predict pack per contrast (05_26: contrast-qualified PACK_DIR, whole node, all 11 methods x 3 folds);
#  4. queue the eval job (06_08) with --dependency=afterany on those pack jobs.
#   bash 05_28_tamia_post_training.sh [--check]     (--check: only report training completeness)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
STATUS="${SCRATCH}/pansegdata/post_training_status.txt"; RUNIDS="${SCRATCH}/pansegdata/_packruns/RUN_IDS.tsv"
mkdir -p "${SCRATCH}/pansegdata"; : > "${STATUS}"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${STATUS}"; }
[ -f "${RUNIDS}" ] || { say "ERROR: ${RUNIDS} missing (training was never launched with 04_23)"; exit 2; }

missing=0
for C in t1wce t2w; do
    ( if [ "${C}" = t2w ]; then source "${HERE}/../00_utils/env_t2w.sh"; else source "${HERE}/../00_utils/env.sh"; fi
      source "${ROOT}/scripts/cluster/tamia_env_pansegdata.sh"
      source "${ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
      pin="$(roster_pin_file)"; mkdir -p "$(dirname "$pin")"; : > "${pin}.new"
      bad=0
      for w in "${HERE}"/05_*_predict_${C}_*.sh; do
          case "$w" in *run_all*) continue;; esac
          roster_parse_wrapper "$w" || exit 3
          if [ "${W_METHOD}" = "auglabAug_v26_6_2_train050_val100" ]; then      # DualVal's val100 mirror = the val000 run id with _val100_
              base="$(awk -F'\t' -v c="${C}" '$1==c && $2=="auglabAug_v26_6_2_train050_val000" {print $3}' "${RUNIDS}")"; rid="${base/_val000_/_val100_}"
          else rid="$(awk -F'\t' -v c="${C}" -v m="${W_METHOD}" '$1==c && $2==m {print $3}' "${RUNIDS}")"; fi
          [ -n "${rid}" ] || { echo "  [${C}] no RUN_ID for ${W_METHOD} in RUN_IDS.tsv"; bad=$((bad+1)); continue; }
          for F in 0 1 2; do
              ls "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${C}/${W_CATEGORY}/${rid}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${F}/checkpoint_final.pth >/dev/null 2>&1 \
                  || { echo "  [${C}] MISSING checkpoint_final: ${W_METHOD} (${rid}) fold ${F}"; bad=$((bad+1)); }
          done
          printf '%s\t%s\t%s\n' "${W_METHOD}" "${W_CATEGORY}" "${rid}" >> "${pin}.new"
      done
      if [ "${bad}" = 0 ]; then mv "${pin}.new" "${pin}"; echo "  [${C}] all runs complete; pinned $(wc -l < "${pin}") methods -> ${pin}"; else rm -f "${pin}.new"; fi
      exit "${bad}" ) 2>&1 | tee -a "${STATUS}"
    rc=${PIPESTATUS[0]}; missing=$((missing + rc))
done
if [ "${missing}" != 0 ]; then say "INCOMPLETE TRAINING: ${missing} problems above. Nothing predicted. Resume/fix, then re-run this script."; exit 2; fi
say "training complete for both contrasts; pins written"
[ "${1:-}" = "--check" ] && exit 0

pack_ids=""
for C in t1wce t2w; do
    out="$( if [ "${C}" = t2w ]; then source "${HERE}/../00_utils/env_t2w.sh"; else source "${HERE}/../00_utils/env.sh"; fi
            source "${ROOT}/scripts/cluster/tamia_env_pansegdata.sh"
            bash "${HERE}/05_26_tamia_pack_predict.sh" "${C}" 2>&1 )"; rc=$?
    echo "${out}" | tail -6 | tee -a "${STATUS}"
    [ "${rc}" = 0 ] || { say "ERROR: predict pack for ${C} failed to record/submit (rc=${rc})"; exit 3; }
    pack_ids="${pack_ids}:$(echo "${out}" | grep -o 'Submitted batch job [0-9]*' | awk '{print $4}' | paste -sd: -)"
done
pack_ids="${pack_ids#:}"; pack_ids="${pack_ids//::/:}"
say "predict packs submitted: ${pack_ids}"
source "${HERE}/../00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_pansegdata.sh"
RUN_JOB_DEPENDENCY="afterany:${pack_ids}" run_job --name pansegdata_eval_all --gpus 0 --cpus 8 --mem 32G --time 05:00:00 \
    --log "${SCRATCH}/pansegdata/eval_stage.log" -- bash "${ROOT}/benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/06_evaluate/06_08_tamia_eval_stage.sh"
say "eval job queued (afterany:${pack_ids}); log ${SCRATCH}/pansegdata/eval_stage.log"
