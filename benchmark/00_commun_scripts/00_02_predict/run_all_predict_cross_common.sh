#!/usr/bin/env bash
# Shared "predict a whole SOURCE roster on a companion's test items" driver (eval-only companions; cross mode of predict_common.sh).
# NOT invoked directly: the companion's 05_02_run_all_predict.sh sources env.sh and then this. For every source TRAINING CONTRAST and every method
# pinned in the SOURCE's roster_run_ids.tsv it resolves the run dir, reads the trainer class + dataset id from that run dir, and runs the companion's
# predict shim (PREDICT_SHIM = 05_01_predict_<src>_common.sh, which sets PREDICT_MODE=cross and sources predict_common.sh) once per method.
#   SOURCE_PREFIX     env-block prefix of the source task (e.g. ISPY2)         SOURCE_CONTRASTS   its training contrasts ("t1wce t2w")
#   PREDICT_SHIM      companion's cross predict shim
# Optional env: ROSTER_ONLY="m1 m2", ROSTER_SKIP_MISSING=1 (source run dir absent), CHECKPOINT (forwarded), RUN_JOB_PACK_DIR (+ ROSTER_PACK_SUBMIT=1) for
# TamIA whole-node packs. Items default to the shim's PREDICT_ITEMS_DEFAULT.
set -euo pipefail
source "${PROJECT_ROOT:?source env.sh first}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
: "${SOURCE_PREFIX:?}" "${SOURCE_CONTRASTS:?}" "${PREDICT_SHIM:?}"
log() { echo "[$(date '+%H:%M:%S')] predict_cross_roster(${SOURCE_PREFIX}): $*"; }
n_ok=0; n_skip=0
for c in ${SOURCE_CONTRASTS}; do
    pins="$(roster_src_pin_file "${SOURCE_PREFIX}" "${c}")"
    [ -f "${pins}" ] || { echo "ERROR: no source roster pins at ${pins} (source task not finished/pinned: roster-era sources get it from their 05_28 controller; older sources: bootstrap_source_pins.py)" >&2; exit 1; }
    while IFS=$'\t' read -r m cat rid _rest; do
        [ -n "${m}" ] || continue
        if [ -n "${ROSTER_ONLY:-}" ] && ! [[ " ${ROSTER_ONLY} " == *" ${m} "* ]]; then continue; fi
        rd="$(roster_src_run_dir "${SOURCE_PREFIX}" "${c}" "${cat}" "${rid}")"
        if ! roster_run_trainer_id "${rd}"; then
            [ "${ROSTER_SKIP_MISSING:-0}" = "1" ] && { log "SKIP ${c}/${m}: no trained model under ${rd}"; n_skip=$((n_skip+1)); continue; }
            echo "ERROR: no trained model (Dataset*/<Trainer>__nnUNetPlans__3d_fullres) under ${rd}" >&2; exit 1
        fi
        log "── ${c}/${m} (${cat}) ${rid} | ${RUN_TRAINER} Dataset${RUN_DATASET_ID} ──"
        (
            export "${SOURCE_PREFIX}_TRAINING_CONTRAST=${c}" "${SOURCE_PREFIX}_DATASET_ID=${RUN_DATASET_ID}"
            METHOD="${m}"; TRAINER="${RUN_TRAINER}"; CATEGORY="${cat}"
            source "${PREDICT_SHIM}" "${rid}" all
        )
        n_ok=$((n_ok+1))
    done < "${pins}"
done
log "done: ${n_ok} runs, ${n_skip} skipped"
if [ -n "${RUN_JOB_PACK_DIR:-}" ] && [ "${ROSTER_PACK_SUBMIT:-0}" = "1" ]; then
    PACK_GPU_TYPE="${PACK_GPU_TYPE:-h100}" PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-02:00:00}" PACK_CHAIN="${PACK_CHAIN:-1}" \
    PACK_JOB_NAME="${PACK_JOB_NAME:-${DATASET_NAME}_predict_cross}" bash "${PROJECT_ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${RUN_JOB_PACK_DIR}"
fi
