#!/usr/bin/env bash
# Shared "evaluate a whole roster" driver, companion of 00_02_predict/run_all_predict_common.sh. NOT invoked directly: a
# per-contrast launcher sources env.sh, sets METHOD_SCRIPTS (the SAME predict wrappers) and EVAL_SCRIPT (the dataset's
# 06_01_evaluate_run.sh), then sources this. Each method's RUN_ID/CATEGORY come from the roster (roster_lib.sh) — no timestamps
# in any launcher. Ladder rungs (ROSTER_ABLATION_METHODS) are evaluated with METRICS_SUBDIR=ablations automatically.
# Optional env: ROSTER_SKIP_MISSING=1, ROSTER_ONLY="m1 m2", CKPT_TAG (forwarded).
set -euo pipefail
source "${PROJECT_ROOT:?source env.sh first}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
: "${EVAL_SCRIPT:?}"; [ -f "${EVAL_SCRIPT}" ] || { echo "ERROR: EVAL_SCRIPT not found: ${EVAL_SCRIPT}" >&2; exit 1; }
[ -n "${METHOD_SCRIPTS+x}" ] && [ "${#METHOD_SCRIPTS[@]}" -gt 0 ] || { echo "ERROR: METHOD_SCRIPTS not set" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] evaluate_roster(${TRAINING_CONTRAST}): $*"; }
n=0
for s in "${METHOD_SCRIPTS[@]}"; do
    roster_parse_wrapper "$s"
    if [ -n "${ROSTER_ONLY:-}" ] && ! [[ " ${ROSTER_ONLY} " == *" ${W_METHOD} "* ]]; then continue; fi
    if ! rid="$(roster_resolve "${W_METHOD}" "${W_CATEGORY}")"; then
        [ "${ROSTER_SKIP_MISSING:-0}" = "1" ] && { log "SKIP ${W_METHOD}: no run"; continue; }
        echo "ERROR: no run for ${W_METHOD}/${W_CATEGORY}" >&2; exit 1
    fi
    sub=""; roster_is_ablation "${W_METHOD}" && sub="ablations"
    log "── ${W_METHOD} (${W_CATEGORY}) ${rid}${sub:+ → ${sub}/} ──"
    METRICS_SUBDIR="${sub}" bash "${EVAL_SCRIPT}" "${rid}" "${W_CATEGORY}" all
    n=$((n+1))
done
log "evaluated ${n} runs"
