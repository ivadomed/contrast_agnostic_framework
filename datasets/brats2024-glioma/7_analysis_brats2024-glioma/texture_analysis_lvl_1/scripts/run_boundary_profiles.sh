#!/usr/bin/env bash
# H7 boundary-profile pipeline, sharded across N parallel CPU-only Vulcan Slurm jobs, followed
# by a dependent summary job. Each shard writes incrementally and skips patients already done,
# so a rerun only recomputes what's missing. Usage: bash run_boundary_profiles.sh [world_size]
set -euo pipefail

WORLD_SIZE="${1:-4}"
LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Submitting ${WORLD_SIZE} boundary-profile shard jobs =="
JOB_IDS=()
for ((r=0; r<WORLD_SIZE; r++)); do
    out="$(run_job --name "bprof_shard${r}" --gpus 0 --cpus 4 --mem 16G --time 00:40:00 \
        --log "${LOGDIR}/bprof_shard${r}.log" -- \
        "${PY}" "${S}/compute_boundary_profiles.py" --rank "${r}" --world-size "${WORLD_SIZE}")"
    echo "${out}"
    jid="$(echo "${out}" | awk '{print $4}')"
    JOB_IDS+=("${jid}")
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
echo "== Submitting summary job, depends on: ${DEP} =="
RUN_JOB_DEPENDENCY="${DEP}" run_job --name bprof_summary --gpus 0 --cpus 4 --mem 16G --time 00:40:00 \
    --log "${LOGDIR}/bprof_summary.log" -- \
    "${PY}" "${S}/boundary_vs_fill_swap_summary.py"

echo "Shards: ${JOB_IDS[*]}"
echo "Watch: squeue -u \$USER ; tail -f ${LOGDIR}/bprof_summary.log"
