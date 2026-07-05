#!/usr/bin/env bash
# Histogram-manifold coverage (Pillar 2) — 31-class pipeline, dispatched through run_job.
# All features re-extracted FRESH on this machine in the 31-class parcellation (matches
# Pillar 1); real uses each scan's own per-modality SynthSeg (FreeSurfer→31), synth uses
# Dataset031 labelsTr directly.
#   1a/1b extract real + synth (parallel Slurm jobs)
#   2     PRDC + Vendi vs balanced real, subject bootstrap CIs
#   3     bar plot (PRDC/Vendi) + PCA/UMAP manifold plots
# Nothing runs on the login node.
#
# Usage:  bash run_coverage_lvl1.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

PY="${PROJECT_ROOT}/.venv/bin/python"
OUT="${HERE}/../outputs"
REAL_CSV="${OUT}/real_regional_hist31.csv"
SYNTH_CSV="${OUT}/synth_regional_hist31.csv"
LOGDIR="${OUT}/logs"; mkdir -p "${LOGDIR}"

echo "[1/3] extract 31-class features (real + synth, parallel; skip if CSV exists)"
[ -f "${REAL_CSV}" ] || run_job --name hcov_real31 --gpus 0 --slot 0 --cpus 32 --mem 64G \
    --log "${LOGDIR}/extract_real31.log" --wait -- \
    "${PY}" "${HERE}/extract_regional_hist31.py" --mode real \
        --output-csv "${REAL_CSV}" --n-workers 32 &
[ -f "${SYNTH_CSV}" ] || run_job --name hcov_synth31 --gpus 0 --slot 0 --cpus 16 --mem 32G \
    --log "${LOGDIR}/extract_synth31.log" --wait -- \
    "${PY}" "${HERE}/extract_regional_hist31.py" --mode synth \
        --output-csv "${SYNTH_CSV}" --n-workers 16 &
wait
echo "  extraction done ($([ -f "${REAL_CSV}" ] && echo real✓) $([ -f "${SYNTH_CSV}" ] && echo synth✓))"

echo "[2/3] compute PRDC + Vendi (balanced real, bootstrap CIs)"
run_job --name hcov_compute --gpus 0 --slot 0 --cpus 8 --mem 32G \
    --log "${LOGDIR}/compute.log" --wait -- \
    "${PY}" "${HERE}/compute_coverage_metrics.py" \
        --real-csv "${REAL_CSV}" --synth-csv "${SYNTH_CSV}" --output-dir "${OUT}"

echo "[3/3] plots (bars + PCA/UMAP)"
run_job --name hcov_bars --gpus 0 --slot 0 --cpus 2 --mem 8G \
    --log "${LOGDIR}/plot_bars.log" --wait -- \
    "${PY}" "${HERE}/plot_coverage_metrics.py" --output-dir "${OUT}" &
run_job --name hcov_manif --gpus 0 --slot 0 --cpus 8 --mem 32G \
    --log "${LOGDIR}/plot_manifold.log" --wait -- \
    "${PY}" "${HERE}/plot_manifold.py" --real-csv "${REAL_CSV}" \
        --synth-csv "${SYNTH_CSV}" --output-dir "${OUT}" &
wait
echo "DONE → ${OUT}"
