#!/usr/bin/env bash
# Queue the post-training controller (05_29) behind ALL pansegdata training fold jobs currently in the Slurm queue (afterany: it fires when the last fold ends,
# successful or not; the controller itself refuses to predict on incomplete training). Run right after 04_28 (it needs all 60 fold jobs queued).
#   bash 05_30_killarney_queue_predict_eval.sh            # queue it
#   bash 05_30_killarney_queue_predict_eval.sh --check    # only show what it would wait for
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
unset RUN_JOB_GPU_TYPE; export RUN_JOB_GPU_TYPE=""     # the controller is CPU-only
source "${HERE}/../00_utils/env.sh"
ids="$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^fold[0-9]_pansegdata_/ {print $1}' | paste -sd: -)"
n=$(echo "${ids}" | tr ':' '\n' | grep -c . || true)
echo "[queue] training fold jobs in the queue: ${n} (expect 60)"
[ "${n}" = 60 ] || { [ "${FORCE:-0}" = 1 ] || { echo "ERROR: expected 60 pansegdata fold jobs, found ${n} (FORCE=1 to override)" >&2; exit 1; }; }
[ "${1:-}" = "--check" ] && { echo "[queue] would wait for: ${ids}"; exit 0; }
mkdir -p "${RESULTS_DIR}/post_training"
RUN_JOB_DEPENDENCY="afterany:${ids}" run_job --name pansegdata_post_training --gpus 0 --cpus 2 --mem 8G --time 06:00:00 \
    --log "${RESULTS_DIR}/post_training/controller.log" -- bash "${HERE}/05_29_killarney_post_training.sh"
echo "[queue] controller queued behind ${n} training jobs; log ${RESULTS_DIR}/post_training/controller.log ; status file ${RESULTS_DIR}/post_training/post_training_status.txt"
