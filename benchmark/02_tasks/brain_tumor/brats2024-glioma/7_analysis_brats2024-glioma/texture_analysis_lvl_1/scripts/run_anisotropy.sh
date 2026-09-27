#!/usr/bin/env bash
# Anisotropy analysis driver (t1n-is-2D-thick-slice hypothesis). CPU-only, Vulcan Slurm via
# run_job (never on the login node, never on TamIA — user explicitly asked for Vulcan).
# Usage: bash run_anisotropy.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Step 0: anisotropy metric self-test (sanity) =="
run_job --name aniso_sanity --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/anisotropy_sanity.log" -- \
    "${PY}" "${S}/compute_anisotropy.py" --sanity
cat "${LOGDIR}/anisotropy_sanity.log"
if ! grep -q "SANITY PASSED" "${LOGDIR}/anisotropy_sanity.log"; then
    echo "Sanity check FAILED — stopping before computing real data." >&2
    exit 1
fi

echo "== Step 0b: axis-code check (which array axis is S-I / A-P / L-R) =="
run_job --name aniso_axcodes --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/anisotropy_axcodes.log" -- \
    "${PY}" "${S}/compute_anisotropy.py" --axcodes-only
cat "${LOGDIR}/anisotropy_axcodes.log"

echo "== Step 1: per-patient anisotropy computation (single shard, all 70 patients) =="
run_job --name aniso_main --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/anisotropy_main.log" -- \
    "${PY}" "${S}/compute_anisotropy.py"
cat "${LOGDIR}/anisotropy_main.log"

echo "== Step 2: merge shards =="
run_job --name aniso_merge --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/anisotropy_merge.log" -- \
    "${PY}" "${S}/compute_anisotropy.py" --merge
cat "${LOGDIR}/anisotropy_merge.log"

echo "== Step 3: summary tables + plots =="
run_job --name aniso_summary --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/anisotropy_summary.log" -- \
    "${PY}" "${S}/anisotropy_summary.py"
cat "${LOGDIR}/anisotropy_summary.log"
