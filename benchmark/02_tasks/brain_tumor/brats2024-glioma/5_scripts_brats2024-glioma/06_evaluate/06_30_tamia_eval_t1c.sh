#!/usr/bin/env bash
# Queue the CPU evaluation jobs for the brats2024-glioma T1c-trained runs, natively
# dependent on the predict job (05_predict/05_81_tamia_pack_predict_t1c.sh). Run ON tamia.
# One CPU job per run (06_31_t1c_eval_job.sh); each audits its own output (70 cases x 4 contrasts
# x folds 0-2) so truncated predictions fail loudly instead of yielding silently partial metrics.
#
# Usage:
#   bash 06_30_tamia_eval_t1c.sh <main|rung4|srcsm> afterany:<predict JOBID>     # or "none" if predictions already exist
# `afterany` (not afterok) on purpose: the eval audit is the correctness gate, and afterok would
# leave the jobs stuck in DependencyNeverSatisfied if a single predict fold-command failed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HERE_DIR="$(cd "${HERE}/.." && pwd)"
GROUP="${1:?usage: 06_30_tamia_eval_t1c.sh <main|rung4|srcsm> afterany:<jobid>}"
DEPENDENCY="${2:?dependency required, e.g. afterany:12345, or none}"
DEP_ARGS=(); [ "${DEPENDENCY}" = "none" ] || DEP_ARGS=(--dependency="${DEPENDENCY}")
source "${HERE_DIR}/00_utils/t1c_runs.sh"
LOGDIR="/scratch/p/paulh/brats2024-glioma/_packruns/t1c_eval_${GROUP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

submit() {   # CPU-only sbatch on TamIA is inconsistent about --partition (see the project notes): try bare, retry with explicit
  local out
  if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
  echo "  (bare sbatch failed — retrying with --partition=cpubase_bynode_b1)" >&2
  sbatch --parsable --partition=cpubase_bynode_b1 "$@" 2>/dev/null
}

for row in "${T1C_RUNS[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"
  [ "${g}" = "${GROUP}" ] || continue
  jid="$(submit "${DEP_ARGS[@]}" --job-name="brats_t1c_eval_${name}" --time=03:00:00 \
        --cpus-per-task=16 --mem=96G --output="${LOGDIR}/eval_${name}_%j.out" \
        --export="ALL,RUN_ID=${run},CATEGORY=${cat},METRICS_SUBDIR=${msub},HERE_DIR=${HERE_DIR}" \
        "${HERE}/06_31_t1c_eval_job.sh")"
  echo "[t1c-eval] ${name}: job ${jid} (dep ${DEPENDENCY}) -> ${cat}_${run} ${msub:+[${msub}]}"
done
echo "[t1c-eval] logs: ${LOGDIR}"
