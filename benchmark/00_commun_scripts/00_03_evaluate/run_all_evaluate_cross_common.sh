#!/usr/bin/env bash
# Shared "evaluate a whole SOURCE roster on a companion's items" driver, companion of 00_02_predict/run_all_predict_cross_common.sh. NOT invoked directly:
# the companion's 06_06_run_all_eval.sh sources env.sh, sets SOURCE_PREFIX / SOURCE_CONTRASTS / EVAL_SCRIPT (its 06_01_evaluate_run.sh), then sources this.
# Ladder rungs (ROSTER_ABLATION_METHODS) automatically get LADDER=1 (-> .../ablations/<item>/). Optional env: ROSTER_ONLY, ROSTER_SKIP_MISSING, CKPT_TAG, EVAL_INLINE.
set -euo pipefail
source "${PROJECT_ROOT:?source env.sh first}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
: "${SOURCE_PREFIX:?}" "${SOURCE_CONTRASTS:?}" "${EVAL_SCRIPT:?}"; [ -f "${EVAL_SCRIPT}" ] || { echo "ERROR: EVAL_SCRIPT not found: ${EVAL_SCRIPT}" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] evaluate_cross_roster(${SOURCE_PREFIX}): $*"; }
n=0
for c in ${SOURCE_CONTRASTS}; do
    pins="$(roster_src_pin_file "${SOURCE_PREFIX}" "${c}")"; [ -f "${pins}" ] || { echo "ERROR: no source roster pins at ${pins}" >&2; exit 1; }
    while IFS=$'\t' read -r m cat rid _rest; do
        [ -n "${m}" ] || continue
        if [ -n "${ROSTER_ONLY:-}" ] && ! [[ " ${ROSTER_ONLY} " == *" ${m} "* ]]; then continue; fi
        lad=0; roster_is_ablation "${m}" && lad=1
        if [ ! -d "${PREDICTIONS_ROOT}/$(roster_src "${SOURCE_PREFIX}" MODEL_TYPE)/${c}/${cat}/${rid}" ]; then
            [ "${ROSTER_SKIP_MISSING:-0}" = "1" ] && { log "SKIP ${c}/${m}: no predictions yet"; continue; }
            echo "ERROR: no predictions for ${c}/${m} (${rid}) -- run the predict launcher first" >&2; exit 1
        fi
        log "── ${c}/${m} (${cat}) ${rid} ladder=${lad} ──"
        LADDER="${lad}" bash "${EVAL_SCRIPT}" "${rid}" "${cat}" "${c}" all
        n=$((n+1))
    done < "${pins}"
done
log "evaluated ${n} runs"
