#!/usr/bin/env bash
# Shared "evaluate a non-default checkpoint" driver — companion to
# 00_02_predict/run_all_predict_checkpoint_sweep.sh. NOT invoked directly. A per-modality
# launcher sources this after sourcing env.sh and setting:
#   EVAL_SCRIPT   path to this dataset's per-fold evaluate entry point (its 06_01_evaluate_run.sh
#                 for brats/chaos/open-ms, or 06_01_evaluate_testset.sh for on-harmony, which
#                 combines predict+eval per fold -- see below).
#   EVAL_ARGS     bash array, one element per method, each the exact positional-arg string to
#                 pass to EVAL_SCRIPT after its RUN_ID is already embedded, e.g.
#                 "mydataset_t1n_baseline_20260101_000000" (brats/chaos, CATEGORY auto-detected)
#                 or "open-ms_flair_baseline_20260101_000000 auglab" (open-ms needs CATEGORY
#                 explicit -- it does NOT auto-detect, see that script's header).
# Required env: CKPT_TAG ("best" | "final") -- passed straight through to EVAL_SCRIPT for
#   brats/chaos/open-ms (their own CKPT_TAG convention: "best" = original flat paths
#   unchanged, anything else = sibling fold{F}/<tag>/<item> + a sibling <cat>_<run>_<tag>
#   metrics dir). On-harmony's 06_01_evaluate_testset.sh predates CKPT_TAG and instead reads
#   CHECKPOINT directly (its predict+eval are one combined step) -- for that one dataset,
#   the per-modality launcher must translate CKPT_TAG into CHECKPOINT itself, e.g.:
#     CHECKPOINT="checkpoint_${CKPT_TAG}.pth"
#   before sourcing this driver, rather than this shared driver guessing dataset-specific
#   filename conventions.
# Optional env: DATASET_ID -- forward explicitly when evaluating a non-default-contrast
#   modality (brats t2w needs DATASET_ID=052, chaos t2spir needs 61 -- both scripts otherwise
#   default to their FIRST-added contrast's dataset id; confirmed the hard way, see git history).
set -euo pipefail

: "${CKPT_TAG:?CKPT_TAG must be set explicitly (best|final) -- no silent default}"
: "${EVAL_SCRIPT:?EVAL_SCRIPT must be set to the dataset per-fold evaluate entry point}"
export CKPT_TAG
[ -f "${EVAL_SCRIPT}" ] || { echo "ERROR: EVAL_SCRIPT not found: ${EVAL_SCRIPT}" >&2; exit 1; }

log() { echo "[$(date '+%H:%M:%S')] evaluate_checkpoint_sweep: $*"; }

if [ -z "${EVAL_ARGS+x}" ] || [ "${#EVAL_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: EVAL_ARGS array not set — a per-modality launcher must define it." >&2; exit 1
fi

log "CKPT_TAG='${CKPT_TAG}'${DATASET_ID:+ DATASET_ID=${DATASET_ID}}  |  $(basename "${EVAL_SCRIPT}")  |  ${#EVAL_ARGS[@]} runs"
for args in "${EVAL_ARGS[@]}"; do
    log "── evaluating ${args} ──"
    bash "${EVAL_SCRIPT}" ${args}
    log "── ${args} done ──"
done
log "all ${#EVAL_ARGS[@]} runs evaluated @ CKPT_TAG=${CKPT_TAG}."
