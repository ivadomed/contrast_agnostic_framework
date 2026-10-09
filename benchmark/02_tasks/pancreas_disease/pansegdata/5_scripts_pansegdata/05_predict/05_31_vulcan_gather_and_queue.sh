#!/usr/bin/env bash
# VULCAN: gather ALL pansegdata training results in this checkout and queue the post-training controller (05_29: verify checkpoint_final x 3 folds for all 22 roster runs, refuse if incomplete,
# pin, predict both contrasts, queue the eval stage 06_09) here. Needed because training is split (2026-10-05): ladder rungs on Vulcan L40S, headline runs on Killarney H100.
# Idempotent: re-run it whenever more Killarney runs have finished; it only copies what is new. Run it once everything is done (the controller itself refuses on incomplete training).
#   bash 05_31_vulcan_gather_and_queue.sh [--no-queue]
# CODE PARITY NOTE: the rung runs trained here with the PINNED AugLab (Killarney 7b761b5, see 04_30_vulcan_rungs.sh); PREDICTION does not use AugLab augmentations.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
unset RUN_JOB_GPU_TYPE; export RUN_JOB_GPU_TYPE=""      # the controller is CPU-only; it sets l40s itself for the predict jobs
source "${HERE}/../00_utils/env.sh"
KILL="killarney.alliancecan.ca"; KREPO="/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project"
KRES="${KREPO}/benchmark/02_tasks/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/"
DST="${PREDICTIONS_ROOT}/${MODEL_TYPE}/"; mkdir -p "${DST}"
echo "[gather] Killarney still has fold jobs queued/running: $(ssh -o BatchMode=yes "${KILL}" 'bash -lc "squeue -u \$USER -h -o %j | grep -c ^fold || true"')"
echo "[gather] pulling Killarney results -> ${DST} (no checkpoint_latest, no stale pin files)"
rsync -a --partial --exclude checkpoint_latest.pth --exclude roster_run_ids.tsv "${KILL}:${KRES}" "${DST}"
n_final=$(find "${DST}" -name checkpoint_final.pth | wc -l)
echo "[gather] checkpoint_final.pth files now in this checkout: ${n_final}  (expect 66 = 11 roster runs x 3 folds x 2 contrasts, the OURS val100 mirror counting as its own run; the controller does the exact per-run check)"
[ "${1:-}" = "--no-queue" ] && exit 0
ids="$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^fold[0-9]_pansegdata_/ {print $1}' | paste -sd: -)"
dep=""; [ -n "${ids}" ] && dep="afterany:${ids}"
mkdir -p "${RESULTS_DIR}/post_training"
echo "[gather] queueing the controller here (dependency: ${dep:-none, Vulcan has no pansegdata fold jobs left})"
RUN_JOB_DEPENDENCY="${dep}" run_job --name pansegdata_post_training --gpus 0 --cpus 2 --mem 8G --time 06:00:00 \
    --log "${RESULTS_DIR}/post_training/controller.log" -- bash "${HERE}/05_29_killarney_post_training.sh"
echo "[gather] status file: ${RESULTS_DIR}/post_training/post_training_status.txt ; if it says INCOMPLETE, finish/resume the missing folds, rerun this script."
