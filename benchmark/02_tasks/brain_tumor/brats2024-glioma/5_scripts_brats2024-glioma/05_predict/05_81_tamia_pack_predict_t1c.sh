#!/usr/bin/env bash
# TAMIA whole-node PREDICT job for the brats2024-glioma T1c-trained runs. Run ON tamia.
# Run table lives in 00_utils/t1c_runs.sh; the job body is 05_82_t1c_predict_job.sh.
#
# Usage:
#   bash 05_81_tamia_pack_predict_t1c.sh main                       # the 9 Pack A+B runs
#   bash 05_81_tamia_pack_predict_t1c.sh srcsm afterany:<PackC last link>  # srcsm, queued natively behind
#                                                                   # the end of its training chain
# Prints the submitted job id (also written to <PACK_DIR>/JOBID) so an eval job can depend on it
# (06_evaluate/06_30_tamia_eval_t1c.sh). PACK_DIR is contrast- AND group-qualified (t1c_predict_*).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HERE_DIR="$(cd "${HERE}/.." && pwd)"          # …/5_scripts_brats2024-glioma
GROUP="${1:?usage: 05_81_tamia_pack_predict_t1c.sh <main|rung4|srcsm> [afterany:JOBID]}"
DEPENDENCY="${2:-}"
case "${GROUP}" in
  main)  TIME="06:00:00" ;;    # 27 fold-commands (9 runs x 3 folds x 4 contrasts x 70 cases), ~7-8 per GPU
  rung4|srcsm) TIME="02:30:00" ;;   # 3 fold-commands, one per GPU
  *) echo "unknown group '${GROUP}'" >&2; exit 2 ;;
esac

PACK_DIR="/scratch/p/paulh/brats2024-glioma/_packruns/t1c_predict_${GROUP}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"
DEP_ARGS=(); [ -n "${DEPENDENCY}" ] && DEP_ARGS=(--dependency="${DEPENDENCY}")
JOBID="$(sbatch --parsable "${DEP_ARGS[@]}" --job-name="brats_t1c_predict_${GROUP}" --time="${TIME}" \
    --output="${PACK_DIR}/predict_job_%j.out" \
    --export="ALL,GROUP=${GROUP},PACK_DIR=${PACK_DIR},HERE_DIR=${HERE_DIR}" \
    "${HERE}/05_82_t1c_predict_job.sh")"
echo "${JOBID}" > "${PACK_DIR}/JOBID"
echo "[t1c-predict] group=${GROUP} job=${JOBID} time=${TIME} dependency='${DEPENDENCY}' PACK_DIR=${PACK_DIR}"
