#!/usr/bin/env bash
# Builds the 4 "see the phenomenon" figures for the T2w<->T1n / mirror T1n<->T2f edema
# real-fill-vs-noise-fill story. CPU-only, small volumes, few patients — runs as one short
# Vulcan Slurm job via run_job (never on the login node, never on TamIA).
# Usage: bash run_see_edema_texture_figures.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

echo "== Building see_edema_texture_gallery / see_fingerprints / see_predictions_* figures =="
run_job --name see_figs --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${LOGDIR}/see_edema_texture_figures.log" -- \
    "${PY}" "${S}/see_edema_texture_figures.py"
tail -n 80 "${LOGDIR}/see_edema_texture_figures.log"
