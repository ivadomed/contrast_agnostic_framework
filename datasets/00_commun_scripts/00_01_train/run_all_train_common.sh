#!/usr/bin/env bash
# Shared "run all methods" driver — launches a dataset's full method suite for one modality.
# NOT invoked directly. A per-modality launcher (e.g. open-ms 04_12_run_all_flair.sh) sources this
# after sourcing env.sh and setting:
#   METHOD_SCRIPTS   bash array of per-method training wrapper paths, in launch order.
# Each wrapper already fans out its own folds via 04_00_common.sh → 00_01_train/train_common.sh, so
# this just launches the wrappers in sequence (sequential avoids GPU contention on the set_slot
# backend; harmless on Slurm, where each wrapper just fires its fold jobs and returns).
#
# Fold policy (CLAUDE.md): caps at 3 folds — exports TRAIN_FOLDS="0 1 2" by default so every wrapper
# inherits it. Override by exporting TRAIN_FOLDS before sourcing.
#
# Optional first arg: --start-from <substring>  → skip wrappers until one matches (resume a suite).
#
# Usage (from a per-modality launcher):
#   METHOD_SCRIPTS=( "$HERE/04_01_train_baseline.sh" ... )
#   source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/run_all_train_common.sh" "$@"
set -euo pipefail

: "${TRAIN_FOLDS:=0 1 2}"; export TRAIN_FOLDS      # 3-fold cap (see CLAUDE.md "FOLD POLICY")

START_FROM=""
if [ "${1:-}" = "--start-from" ]; then START_FROM="${2:?--start-from needs a value}"; shift 2; fi

log() { echo "[$(date '+%H:%M:%S')] run_all: $*"; }

if [ -z "${METHOD_SCRIPTS+x}" ] || [ "${#METHOD_SCRIPTS[@]}" -eq 0 ]; then
    echo "ERROR: METHOD_SCRIPTS array not set — a per-modality launcher must define it." >&2; exit 1
fi

log "TRAIN_FOLDS='${TRAIN_FOLDS}'  |  ${#METHOD_SCRIPTS[@]} methods${START_FROM:+  (start-from: ${START_FROM})}"
started=0; [ -z "${START_FROM}" ] && started=1
for s in "${METHOD_SCRIPTS[@]}"; do
    name="$(basename "$s")"
    if [ "${started}" -eq 0 ]; then
        case "${name}" in *"${START_FROM}"*) started=1 ;; *) log "skip ${name}"; continue ;; esac
    fi
    [ -f "$s" ] || { echo "ERROR: method script not found: $s" >&2; exit 1; }
    log "── launching ${name} ──"
    bash "$s"                                       # fresh RUN_ID per method (no args → auto-timestamp)
    log "── ${name} launched ──"
done
log "all ${#METHOD_SCRIPTS[@]} methods launched (TRAIN_FOLDS='${TRAIN_FOLDS}')."
