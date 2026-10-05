#!/usr/bin/env bash
# Run ON TamIA right after launching training (04_23_tamia_pack_all.sh): queues the WHOLE post-training chain with Slurm dependencies so nothing
# has to be babysat:
#   training pack jobs (pansegdata_pack*)  --afterany-->  05_28 controller (CPU job: verify checkpoints, pin RUN_IDs, record + submit one predict
#   pack per contrast)  --afterany-->  06_08 eval job (CPU job: evaluate every roster run, inline)
# then only the manual tail remains (fetch metrics to Vulcan, 06_05 write configs, 06_07 aggregation) -- see the skill.
#   bash 05_27_tamia_queue_predict_eval.sh [--check]      (--check: just print which jobs it would depend on)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
source "${HERE}/../00_utils/env.sh"
source "${ROOT}/scripts/cluster/tamia_env_pansegdata.sh"
# only THIS user's training pack jobs (the 04_23 names), pending or running
IDS="$(squeue -u "${USER}" -h -o "%i %j" | awk '$2 ~ /^pansegdata_pack[0-9]+$/ {print $1}' | paste -sd: -)"
[ -n "${IDS}" ] || { echo "ERROR: no pansegdata_pack* jobs in the queue -- launch 04_23_tamia_pack_all.sh first (or training already finished: run 05_28 directly)" >&2; exit 1; }
echo "[queue] controller will depend on afterany:${IDS}"
[ "${1:-}" = "--check" ] && exit 0
mkdir -p "${SCRATCH}/pansegdata"
RUN_JOB_DEPENDENCY="afterany:${IDS}" run_job --name pansegdata_post_train --gpus 0 --cpus 2 --mem 8G --time 01:00:00 \
    --log "${SCRATCH}/pansegdata/post_training_controller.log" -- bash "${HERE}/05_28_tamia_post_training.sh"
echo "[queue] queued. Controller log: ${SCRATCH}/pansegdata/post_training_controller.log ; status file: ${SCRATCH}/pansegdata/post_training_status.txt"
