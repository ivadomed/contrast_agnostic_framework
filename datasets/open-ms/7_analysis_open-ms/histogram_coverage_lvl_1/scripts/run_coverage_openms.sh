#!/usr/bin/env bash
# open-ms Pillar-2 histogram coverage — CPU analysis (Stages 2-3), dispatched via run_job.
# open-ms uses [lesion histogram, brain histogram] (2 regions) — NO SynthSeg parcellation;
# uses the labels open-ms actually has (FLAIR lesion dseg + brainmask, all co-registered).
#
# PREREQUISITE (GPU, run first on romane — see ROMANE_PROMPT_openms.md):
#   Stage 1  generate_openms_volumes.py  → data/generated/<method>/<key>/*.nii.gz  (FLAIR+T1w sources)
# This script then does, per source contrast, against the SAME 3-contrast real reference:
#   extract real (once) + synth(source) → per-group Coverage + Vendi → bar/heatmap + PCA/UMAP.
# Reuses the dataset-agnostic on-harmony compute/plot scripts unchanged.
#
# Usage:  bash run_coverage_openms.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../../.." && pwd)"
source "${REPO}/scripts/job_runner/run_job.sh"
PY="${REPO}/.venv/bin/python"
ONH="${REPO}/datasets/on-harmony/7_analysis_on-harmony/histogram_coverage_lvl_1/scripts"   # reuse compute/plot
OUT="${HERE}/../outputs"; LOGDIR="${OUT}/logs"; mkdir -p "${LOGDIR}"
REAL_CSV="${OUT}/real_lesion_overall.csv"

echo "[extract] real (3 contrasts) + synth per source"
[ -f "${REAL_CSV}" ] || run_job --name oms_real --gpus 0 --slot 0 --cpus 16 --mem 16G \
    --log "${LOGDIR}/extract_real.log" --wait -- \
    "${PY}" "${HERE}/extract_lesion_overall_openms.py" --mode real --output-csv "${REAL_CSV}" --n-workers 16
for SRC in FLAIR T1w; do
    CSV="${OUT}/synth_${SRC,,}_lesion_overall.csv"
    [ -f "${CSV}" ] || run_job --name "oms_synth_${SRC}" --gpus 0 --slot 0 --cpus 16 --mem 16G \
        --log "${LOGDIR}/extract_synth_${SRC}.log" --wait -- \
        "${PY}" "${HERE}/extract_lesion_overall_openms.py" --mode synth --source "${SRC}" \
            --output-csv "${CSV}" --n-workers 16 &
done
wait

for SRC in flair t1w; do
    SDIR="${OUT}/${SRC}"; mkdir -p "${SDIR}"
    echo "[compute+plot] source=${SRC} → ${SDIR}"
    run_job --name "oms_cmp_${SRC}" --gpus 0 --slot 0 --cpus 8 --mem 16G \
        --log "${LOGDIR}/compute_${SRC}.log" --wait -- \
        "${PY}" "${ONH}/compute_coverage_metrics.py" \
            --real-csv "${REAL_CSV}" --synth-csv "${OUT}/synth_${SRC}_lesion_overall.csv" --output-dir "${SDIR}"
    run_job --name "oms_bar_${SRC}" --gpus 0 --slot 0 --cpus 2 --mem 8G \
        --log "${LOGDIR}/plot_bar_${SRC}.log" --wait -- \
        "${PY}" "${ONH}/plot_coverage_metrics.py" --output-dir "${SDIR}" &
    run_job --name "oms_man_${SRC}" --gpus 0 --slot 0 --cpus 8 --mem 16G \
        --log "${LOGDIR}/plot_manifold_${SRC}.log" --wait -- \
        "${PY}" "${ONH}/plot_manifold.py" --real-csv "${REAL_CSV}" \
            --synth-csv "${OUT}/synth_${SRC}_lesion_overall.csv" --output-dir "${SDIR}" &
    wait
done
echo "DONE → ${OUT}/{flair,t1w}/"
