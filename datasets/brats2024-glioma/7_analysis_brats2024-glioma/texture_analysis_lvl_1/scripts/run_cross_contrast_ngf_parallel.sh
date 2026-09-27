#!/usr/bin/env bash
# Cross-contrast NGF, sharded across N parallel CPU-only Vulcan Slurm jobs, followed by a
# dependent merge+correlate job. Each shard writes incrementally to its own CSV and skips
# patients already present there, so a rerun of this script only recomputes what's missing.
# Usage: bash run_cross_contrast_ngf_parallel.sh [world_size]
set -euo pipefail

WORLD_SIZE="${1:-4}"
LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Submitting ${WORLD_SIZE} shard jobs =="
JOB_IDS=()
for ((r=0; r<WORLD_SIZE; r++)); do
    out="$(run_job --name "ngf_shard${r}" --gpus 0 --cpus 4 --mem 16G --time 01:00:00 \
        --log "${LOGDIR}/shard${r}.log" -- \
        "${PY}" "${S}/compute_cross_contrast_ngf.py" --device cpu --rank "${r}" --world-size "${WORLD_SIZE}")"
    echo "${out}"
    jid="$(echo "${out}" | awk '{print $4}')"
    JOB_IDS+=("${jid}")
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
echo "== Submitting merge job, depends on: ${DEP} =="
RUN_JOB_DEPENDENCY="${DEP}" run_job --name ngf_merge --gpus 0 --cpus 2 --mem 8G --time 00:15:00 \
    --log "${LOGDIR}/merge.log" -- \
    "${PY}" "${S}/compute_cross_contrast_ngf.py" --device cpu --merge

echo "Shards: ${JOB_IDS[*]}"
echo "Watch: squeue -u \$USER ; tail -f ${LOGDIR}/merge.log"
