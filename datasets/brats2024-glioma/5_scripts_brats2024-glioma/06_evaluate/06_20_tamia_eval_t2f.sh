#!/usr/bin/env bash
# Queue the CPU evaluation jobs for the brats2024-glioma T2f/FLAIR-trained runs, natively
# dependent on the predict job (05_predict/05_61_tamia_pack_predict_t2f.sh). Run ON tamia.
# One CPU job per run (06_21_t2f_eval_job.sh); each audits its own output (70 cases x 4 contrasts
# x folds 0-2) so truncated predictions fail loudly instead of yielding silently partial metrics.
#
# Usage:
#   bash 06_20_tamia_eval_t2f.sh <main|srcsm> afterany:<predict JOBID>
# `afterany` (not afterok) on purpose: the eval audit is the correctness gate, and afterok would
# leave the jobs stuck in DependencyNeverSatisfied if a single predict fold-command failed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HERE_DIR="$(cd "${HERE}/.." && pwd)"
GROUP="${1:?usage: 06_20_tamia_eval_t2f.sh <main|srcsm> afterany:<jobid>}"
DEPENDENCY="${2:?dependency required, e.g. afterany:12345}"
source "${HERE_DIR}/00_utils/t2f_runs.sh"
LOGDIR="/scratch/p/paulh/brats2024-glioma/_packruns/t2f_eval_${GROUP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

submit() {   # CPU-only sbatch on TamIA is inconsistent about --partition (see CLAUDE.md): try bare, retry with explicit
  local out
  if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
  echo "  (bare sbatch failed — retrying with --partition=cpubase_bynode_b1)" >&2
  sbatch --parsable --partition=cpubase_bynode_b1 "$@" 2>/dev/null
}

for row in "${T2F_RUNS[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"
  [ "${g}" = "${GROUP}" ] || continue
  jid="$(submit --dependency="${DEPENDENCY}" --job-name="brats_t2f_eval_${name}" --time=03:00:00 \
        --cpus-per-task=16 --mem=64G --output="${LOGDIR}/eval_${name}_%j.out" \
        --export="ALL,RUN_ID=${run},CATEGORY=${cat},METRICS_SUBDIR=${msub},HERE_DIR=${HERE_DIR}" \
        "${HERE}/06_21_t2f_eval_job.sh")"
  echo "[t2f-eval] ${name}: job ${jid} (dep ${DEPENDENCY}) -> ${cat}_${run} ${msub:+[${msub}]}"
done
echo "[t2f-eval] logs: ${LOGDIR}"
