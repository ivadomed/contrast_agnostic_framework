#!/usr/bin/env bash
# Shared "predict with a non-default checkpoint" driver — re-predicts an already-
# trained method suite's EXISTING RUN_IDs at a given checkpoint (e.g. checkpoint_final.pth
# instead of the usual checkpoint_best.pth), for a checkpoint_best-vs-checkpoint_final
# comparison. NOT invoked directly. A per-modality launcher (e.g. a dataset's
# 05_XX_run_all_predict_checkpoint_final.sh) sources this after sourcing env.sh and setting:
#   METHOD_SCRIPTS   bash array of per-method predict wrapper paths, in launch order.
#   METHOD_RUN_IDS   bash array of EXISTING RUN_IDs (same order/length as METHOD_SCRIPTS) --
#                    unlike training's run_all (which starts a fresh RUN_ID per method), this
#                    re-predicts specific already-trained runs, so the caller must supply them.
# Required env: CHECKPOINT (e.g. "checkpoint_final.pth") -- no default on purpose: a checkpoint
#   sweep must be explicit about which checkpoint it's re-predicting, never silently fall back
#   to predict_common.sh's own "checkpoint_best.pth" default.
# predict_common.sh already routes non-"best" checkpoints to a sibling fold{F}/<tag>/<item>
# path (see its CKPT_SUBDIR logic) -- this driver adds no new path handling, just the loop.
#
# Usage (from a per-modality launcher):
#   METHOD_SCRIPTS=( "$HERE/05_01_predict_baseline.sh" ... )
#   METHOD_RUN_IDS=( "mydataset_t1n_baseline_20260101_000000" ... )
#   export CHECKPOINT=checkpoint_final.pth
#   source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
#
# On tamia, run this with RUN_JOB_PACK_DIR set first (same pack-record mechanism training/
# predict already use) so run_job records instead of submits, then submit the pack with
# scripts/job_runner/run_job_pack_submit.sh -- this driver is backend/pack-mode agnostic,
# it just calls each wrapper in sequence like run_all_train_common.sh does.
set -euo pipefail

: "${CHECKPOINT:?CHECKPOINT must be set explicitly (e.g. checkpoint_final.pth) -- no silent default}"
export CHECKPOINT

log() { echo "[$(date '+%H:%M:%S')] predict_checkpoint_sweep: $*"; }

if [ -z "${METHOD_SCRIPTS+x}" ] || [ "${#METHOD_SCRIPTS[@]}" -eq 0 ]; then
    echo "ERROR: METHOD_SCRIPTS array not set — a per-modality launcher must define it." >&2; exit 1
fi
if [ -z "${METHOD_RUN_IDS+x}" ] || [ "${#METHOD_RUN_IDS[@]}" -ne "${#METHOD_SCRIPTS[@]}" ]; then
    echo "ERROR: METHOD_RUN_IDS must be set, same length as METHOD_SCRIPTS." >&2; exit 1
fi

log "CHECKPOINT='${CHECKPOINT}'  |  ${#METHOD_SCRIPTS[@]} methods"
for i in "${!METHOD_SCRIPTS[@]}"; do
    s="${METHOD_SCRIPTS[$i]}"; rid="${METHOD_RUN_IDS[$i]}"
    name="$(basename "$s")"
    [ -f "$s" ] || { echo "ERROR: method script not found: $s" >&2; exit 1; }
    log "── predicting ${name} (${rid}) @ ${CHECKPOINT} ──"
    bash "$s" "$rid"
    log "── ${name} done ──"
done
log "all ${#METHOD_SCRIPTS[@]} methods predicted @ ${CHECKPOINT}."
