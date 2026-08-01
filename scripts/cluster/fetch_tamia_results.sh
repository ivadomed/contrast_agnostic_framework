#!/usr/bin/env bash
# Pull a dataset's TamIA-produced results back into this (Vulcan) repo checkout.
#
#   bash scripts/cluster/fetch_tamia_results.sh <dataset> [--metrics-only]
#   bash scripts/cluster/fetch_tamia_results.sh kidney-t2w
#
# WHY THIS EXISTS
# The tamia_env_*cross.sh overrides point PREDICTIONS_ROOT at TamIA's $SCRATCH
# (its $PROJECT has a hard file-COUNT quota, and prediction trees are tens of
# thousands of small files). Their header comments used to call predictions
# "large/regenerable", implying scratch was a fine permanent home. That was
# wrong on two counts:
#   1. TamIA $SCRATCH is PURGE-ON-INACTIVITY -- the predictions can silently
#      vanish, after which nothing can be re-inspected without a full re-run.
#   2. "Regenerable" ignores the cost: regenerating means re-running the whole
#      predict pack job just to look at one segmentation.
# Predictions are the primary artifact for inspecting WHY a method scored what
# it scored (overlay vs GT), so they belong on the backed-up $PROJECT
# filesystem alongside every Vulcan-run dataset's predictions (amos, sliver07,
# trusted, ... all have theirs here).
#
# Consequence of the old behaviour, found 2026-08-01: msd-spleen,
# cirrmri-liver AND kidney-t2w all had ZERO predictions on Vulcan while their
# metrics were present -- caught only when someone asked to look at the
# predictions. Run this after ANY TamIA predict/evaluate pack job.
#
# Requires a live TamIA ssh relay from this Vulcan session (Duo 2FA, see
# CLAUDE.md "TamIA"). Verifies exact file counts and refuses to report success
# on a short transfer.
set -euo pipefail

DATASET="${1:?usage: fetch_tamia_results.sh <dataset> [--metrics-only]}"
MODE="${2:-all}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAMIA_HOST="${TAMIA_HOST:-tamia.alliancecan.ca}"
TAMIA_SCRATCH="${TAMIA_SCRATCH:-/scratch/p/paulh}"

LOCAL_RESULTS="${PROJECT_ROOT}/datasets/${DATASET}/8_results_${DATASET}"
REMOTE_RESULTS="${TAMIA_SCRATCH}/${DATASET}/8_results_${DATASET}"

[ -d "${LOCAL_RESULTS}" ] || { echo "ERROR: no local results dir: ${LOCAL_RESULTS}" >&2; exit 1; }

ssh -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA_HOST}" true 2>/dev/null || {
    echo "ERROR: cannot reach ${TAMIA_HOST}." >&2
    echo "  The relay needs an interactive Duo 2FA login from a Vulcan shell:" >&2
    echo "    ssh ${TAMIA_HOST}   # complete Duo; leaves the ControlMaster socket live" >&2
    echo "  Check with: ssh -O check ${TAMIA_HOST}" >&2
    exit 1
}

fetch_subdir() {   # $1 = subdir under 8_results_<ds>/ (e.g. 01_predictions)
    local sub="$1"
    local remote="${REMOTE_RESULTS}/${sub}"

    if ! ssh "${TAMIA_HOST}" "test -d '${remote}'" 2>/dev/null; then
        echo "  [skip] ${sub}: not present on TamIA (${remote})"
        return 0
    fi

    local n_remote
    n_remote=$(ssh "${TAMIA_HOST}" "find '${remote}' -type f | wc -l")
    echo "  [fetch] ${sub}: ${n_remote} files from TamIA"
    [ "${n_remote}" -gt 0 ] || { echo "  [skip] ${sub}: empty on TamIA"; return 0; }

    mkdir -p "${LOCAL_RESULTS}"
    # tar-over-ssh: one stream, far faster than rsync for many small files
    ( cd "${LOCAL_RESULTS}" \
      && ssh "${TAMIA_HOST}" "cd '${REMOTE_RESULTS}' && tar cf - '${sub}'" | tar xf - )

    # Verify every REMOTE file arrived. Do NOT compare total local count:
    # the local tree legitimately holds extra Vulcan-only artifacts (aggregate
    # summaries, significance tables, heatmap PNGs) that never exist on TamIA,
    # so count-equality gives a false failure (hit this on brats-ssa2024: 274
    # local vs 266 remote, transfer was actually complete).
    local missing
    missing=$(ssh "${TAMIA_HOST}" "cd '${remote}' && find . -type f" \
              | while read -r rel; do
                    [ -f "${LOCAL_RESULTS}/${sub}/${rel#./}" ] || echo "${rel}"
                done | head -20)
    if [ -n "${missing}" ]; then
        echo "  [FAIL] ${sub}: files missing locally after transfer:" >&2
        echo "${missing}" | sed 's/^/           /' >&2
        return 1
    fi
    local n_local
    n_local=$(find "${LOCAL_RESULTS}/${sub}" -type f | wc -l)
    echo "  [ok]   ${sub}: all ${n_remote} remote files present (${n_local} local incl. Vulcan-only extras)"
}

echo "[$(date '+%H:%M:%S')] fetching ${DATASET} results from ${TAMIA_HOST}"
rc=0
if [ "${MODE}" = "--metrics-only" ]; then
    fetch_subdir 02_metrics || rc=1
else
    fetch_subdir 01_predictions || rc=1
    fetch_subdir 02_metrics     || rc=1
fi

if [ "${rc}" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ${DATASET}: done, all counts verified"
else
    echo "[$(date '+%H:%M:%S')] ${DATASET}: FAILED -- see errors above" >&2
fi
exit "${rc}"
