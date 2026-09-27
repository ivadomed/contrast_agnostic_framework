#!/usr/bin/env bash
# Cross-contrast NGF pilot driver. CPU-only, single/few patients — runs as one short Vulcan
# Slurm job via run_job (never on the login node, never on TamIA — user explicitly asked for
# Vulcan). Usage: bash run_cross_contrast_ngf.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Step 0: NGF metric self-test =="
run_job --name ngf_pilot_sanity --gpus 0 --cpus 2 --mem 4G --time 00:05:00 --wait \
    --log "${LOGDIR}/sanity.log" -- \
    "${PY}" "${S}/compute_cross_contrast_ngf.py" --sanity --device cpu
cat "${LOGDIR}/sanity.log"

echo "== Step 1: cross-contrast NGF pilot (patient search + compute + ladder comparison) =="
run_job --name ngf_pilot_main --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/main.log" -- \
    "${PY}" "${S}/compute_cross_contrast_ngf.py" --device cpu
cat "${LOGDIR}/main.log"
