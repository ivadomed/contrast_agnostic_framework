#!/usr/bin/env bash
# H6 step-vs-affine-texture: sharded across N parallel CPU-only Vulcan Slurm jobs, followed
# by a dependent summary job (step_affine_vs_fill_swap_summary.py). Each shard writes
# incrementally and skips patients already present, so a rerun only recomputes what's
# missing. Mirrors run_cross_contrast_ngf_parallel.sh's pattern exactly.
# Usage: bash run_step_affine_parallel.sh [world_size]
set -euo pipefail

WORLD_SIZE="${1:-4}"
LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Pre-flight: sanity checks + summary-script import, blocking (--wait) =="
run_job --name step_affine_preflight --gpus 0 --cpus 2 --mem 8G --time 00:10:00 --wait \
    --log "${LOGDIR}/preflight.log" -- \
    bash -c "'${PY}' '${S}/compute_step_affine.py' --sanity && cd '${S}' && '${PY}' -c 'import step_affine_vs_fill_swap_summary'"
echo "Pre-flight OK (see ${LOGDIR}/preflight.log) — submitting shards."

echo "== Submitting ${WORLD_SIZE} step_affine shard jobs =="
JOB_IDS=()
for ((r=0; r<WORLD_SIZE; r++)); do
    out="$(run_job --name "step_affine_shard${r}" --gpus 0 --cpus 4 --mem 16G --time 00:40:00 \
        --log "${LOGDIR}/step_affine_shard${r}.log" -- \
        "${PY}" "${S}/compute_step_affine.py" --rank "${r}" --world-size "${WORLD_SIZE}")"
    echo "${out}"
    jid="$(echo "${out}" | awk '{print $4}')"
    JOB_IDS+=("${jid}")
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
echo "== Submitting summary job, depends on: ${DEP} =="
RUN_JOB_DEPENDENCY="${DEP}" run_job --name step_affine_summary --gpus 0 --cpus 2 --mem 8G --time 00:20:00 \
    --log "${LOGDIR}/step_affine_summary.log" -- \
    "${PY}" "${S}/step_affine_vs_fill_swap_summary.py"

echo "Shards: ${JOB_IDS[*]}"
echo "Watch: squeue -u \$USER ; tail -f ${LOGDIR}/step_affine_summary.log"
