#!/usr/bin/env bash
# open-ms Pillar-1 texture analysis driver (census + NMI), mirrors on-harmony's
# run_texture_lvl1.sh but on Slurm (Killarney) via run_job, and per-source-contrast
# (FLAIR / T1w) the way run_coverage_openms.sh splits open-ms's Pillar-2 coverage analysis.
#
# Steps:
#   0. metric self-test (must PASS)
#   1. generate source-aligned augmented volumes (GPU, 4-shard) — skipped if already populated
#   2. compute census/NMI metrics, per source contrast (GPU, 4-shard each)
#   3. aggregate + plot (reuses on-harmony's dataset-agnostic scripts UNCHANGED)
#
# All Slurm GPU jobs are capped at 45 min walltime so they schedule quickly on Killarney.
# Usage:  bash run_texture_lvl1_openms.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../../.." && pwd)"
COVERAGE_LVL1="${REPO}/datasets/open-ms/7_analysis_open-ms/histogram_coverage_lvl_1"
ONH_TEX="${REPO}/datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/scripts"
S="${LVL1}/scripts"
GENERATED="${LVL1}/../data/generated"   # shared with Pillar-2 coverage analysis, lives one level up
DATA="${LVL1}/outputs/data"
TABLES="${LVL1}/outputs/tables"
PLOTS="${LVL1}/outputs/plots"
LOGDIR="${LVL1}/outputs/logs"
METHODS="palette,synthseg_em,synthseg_noem,auglab_default"
JOB_TIME="00:45:00"
mkdir -p "${DATA}" "${TABLES}" "${PLOTS}" "${LOGDIR}"

source "${REPO}/datasets/open-ms/5_scripts_open-ms/00_utils/env.sh"
PY="${REPO}/.venv/bin/python"
# torch's default thread count (= nproc) causes severe contention on a shared, thread-starved
# login/compute node — same class of issue as the blosc2 gotcha in CLAUDE.md. Cap it.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

echo "== Step 0: metric self-test =="
"${PY}" "${S}/compute_texture_metrics_openms.py" --sanity --device cpu

echo "== Step 1: generate augmented volumes (shared with Pillar-2 coverage; skip if populated) =="
N_EXISTING=$(find "${GENERATED}" -name '*.nii.gz' 2>/dev/null | wc -l)
if [ "${N_EXISTING}" -lt 1000 ]; then
    run_job --name oms_tex_gen --gpus 4 --time "${JOB_TIME}" --log "${LOGDIR}/generate.log" --wait -- \
        bash -c "for R in 0 1 2 3; do \
            CUDA_VISIBLE_DEVICES=\$R ${PY} ${COVERAGE_LVL1}/scripts/generate_openms_volumes.py \
                --rank \$R --world-size 4 --out-root ${GENERATED} & \
        done; wait"
else
    echo "  ${N_EXISTING} volumes already present, skipping generation"
fi

echo "== Step 2: compute metrics, per source contrast (4 GPU shards each) =="
run_source () {   # $1 = SOURCE (FLAIR/T1w), $2 = set-label (flair/t1w)
    local src="$1" label="$2"
    run_job --name "oms_tex_${label}" --gpus 4 --time "${JOB_TIME}" \
        --log "${LOGDIR}/compute_${label}.log" --wait -- \
        bash -c "for R in 0 1 2 3; do \
            CUDA_VISIBLE_DEVICES=\$R ${PY} ${S}/compute_texture_metrics_openms.py \
                --device cuda --rank \$R --world-size 4 --source ${src} --set-label ${label} \
                --generated-root ${GENERATED} --methods ${METHODS} \
                --output-csv ${DATA}/metrics_${label}_rank\${R}.csv & \
        done; wait"
}
run_source FLAIR flair
run_source T1w   t1w

# inline monotone controls (gamma/histeq) once — set=ref, source-only, CPU is enough
run_job --name oms_tex_ref --gpus 1 --time "${JOB_TIME}" --log "${LOGDIR}/compute_ref.log" --wait -- \
    "${PY}" "${S}/compute_texture_metrics_openms.py" --device cuda --source FLAIR \
        --methods "gamma,histeq" --set-label ref --output-csv "${DATA}/metrics_ref_rank0.csv"

echo "  shards: $(ls ${DATA}/metrics_*_rank*.csv 2>/dev/null | wc -l)"

echo "== Step 3: aggregate + plot (on-harmony scripts, unchanged) =="
"${PY}" "${ONH_TEX}/aggregate_texture_metrics.py" \
    --input "${DATA}/metrics_*_rank*.csv" --output-dir "${TABLES}"
"${PY}" "${ONH_TEX}/plot_texture_metrics.py" \
    --input "${DATA}/metrics_*_rank*.csv" --output-dir "${PLOTS}"

echo "== DONE =="
echo "  tables → ${TABLES}   plots → ${PLOTS}"
