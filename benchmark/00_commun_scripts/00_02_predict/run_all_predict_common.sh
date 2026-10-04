#!/usr/bin/env bash
# Shared "predict a whole roster" driver. NOT invoked directly: a per-contrast launcher sources env.sh (or env_<contrast>.sh),
# sets METHOD_SCRIPTS, then sources this. For every predict wrapper it reads METHOD/CATEGORY from the wrapper, resolves the
# trained RUN_ID (roster_lib.sh: pin file, else newest run dir), pins it, and runs `bash <wrapper> <RUN_ID> all`.
#
#   METHOD_SCRIPTS         bash array of this contrast's predict wrappers (headline methods, val100 mirror, ladder rungs)
# Optional env:
#   ROSTER_SKIP_MISSING=1  warn and continue when a method has no trained run yet (default: abort)
#   ROSTER_ONLY="m1 m2"    only these METHOD ids
#   CHECKPOINT             forwarded to the wrappers (e.g. checkpoint_final.pth); resolution is unchanged
#   RUN_JOB_PACK_DIR       pack-record mode (TamIA): wrappers RECORD fold commands instead of submitting. With
#                          ROSTER_PACK_SUBMIT=1 this driver then submits the pack (PACK_GPU_TYPE/PACK_TIME/... as in
#                          scripts/job_runner/run_job_pack_submit.sh).
# Resolved RUN_IDs are written to ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/roster_run_ids.tsv; the evaluate
# driver and config generator read the same file, so predict -> eval -> aggregation can never disagree on which run is which.
set -euo pipefail
source "${PROJECT_ROOT:?source env.sh first}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
[ -n "${METHOD_SCRIPTS+x}" ] && [ "${#METHOD_SCRIPTS[@]}" -gt 0 ] || { echo "ERROR: METHOD_SCRIPTS not set" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] predict_roster(${TRAINING_CONTRAST}): $*"; }

n_ok=0; n_skip=0
for s in "${METHOD_SCRIPTS[@]}"; do
    [ -f "$s" ] || { echo "ERROR: wrapper not found: $s" >&2; exit 1; }
    roster_parse_wrapper "$s"
    if [ -n "${ROSTER_ONLY:-}" ] && ! [[ " ${ROSTER_ONLY} " == *" ${W_METHOD} "* ]]; then continue; fi
    if ! rid="$(roster_resolve "${W_METHOD}" "${W_CATEGORY}")"; then
        if [ "${ROSTER_SKIP_MISSING:-0}" = "1" ]; then log "SKIP ${W_METHOD}: no trained run under ${W_CATEGORY}/"; n_skip=$((n_skip+1)); continue; fi
        echo "ERROR: no trained run for METHOD=${W_METHOD} CATEGORY=${W_CATEGORY} (looked in ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${W_CATEGORY}/). Set ROSTER_SKIP_MISSING=1 to skip." >&2
        exit 1
    fi
    roster_pin "${W_METHOD}" "${W_CATEGORY}" "${rid}"
    log "── ${W_METHOD} (${W_CATEGORY}) → ${rid} ──"
    bash "$s" "${rid}" all
    n_ok=$((n_ok+1))
done
log "done: ${n_ok} methods, ${n_skip} skipped. pins: $(roster_pin_file)"

if [ -n "${RUN_JOB_PACK_DIR:-}" ] && [ "${ROSTER_PACK_SUBMIT:-0}" = "1" ]; then
    log "submitting pack ${RUN_JOB_PACK_DIR}"
    PACK_GPU_TYPE="${PACK_GPU_TYPE:-h100}" PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-02:00:00}" \
    PACK_CHAIN="${PACK_CHAIN:-1}" PACK_JOB_NAME="${PACK_JOB_NAME:-${DATASET_NAME}_predict_${TRAINING_CONTRAST}}" \
        bash "${PROJECT_ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${RUN_JOB_PACK_DIR}"
fi
