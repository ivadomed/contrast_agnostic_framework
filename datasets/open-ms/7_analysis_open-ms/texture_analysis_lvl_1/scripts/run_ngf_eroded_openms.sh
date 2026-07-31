#!/usr/bin/env bash
# open-ms Pillar-1 — NGF on the foreground-INTERIOR (erode x3) ROI.
#
# Why this script exists: `plot_ngf_texture.py` (the headline plot) reads
#   ngf_*_<cond>_eroded_rank*.csv  with  set == "<contrast>_<cond>_eroded"
# i.e. the ERODED / interior variant. Those CSVs were produced on Killarney and are
# NOT present on Vulcan — only the non-eroded `ngf_*_noblur_rank*.csv` are. Computing
# NGF on the whole foreground gives the WRONG ordering (auglab_default appears above
# palette) because auglab_default is largely monotone and so trivially preserves
# gradients AT TISSUE BOUNDARIES; its advantage is boundary-driven. Eroding x3 strips
# that shell and measures genuine interior texture, where palette wins
# (FINDINGS.md: paired Wilcoxon p=7.6e-3 FLAIR / p=1.2e-6 T1w). Same boundary-artifact
# logic that unmasked SynthSeg's hard-coded sigma=0.5 blur (interior 0.077 vs shell 0.147).
#
# Also includes v26_6_2_noisefill_v2, which the census-era METHODS list omitted. That arm
# is palette's OWN partition (K-means + label-remap + Voronoi) with the real-intensity fill
# swapped for noise, so palette-vs-noisefill isolates the FILL as the single variable --
# the input-space half of the ladder's rung-4 -> rung-5 causal step
# (02_ablation_deployed_config_summary.md: OOD Dice 19.55 -> 27.30).
#
# Usage:  bash run_ngf_eroded_openms.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 4 levels: texture_analysis_lvl_1 -> 7_analysis_open-ms -> open-ms -> datasets -> REPO.
# (run_texture_lvl1_openms.sh uses 5, which overshoots to $PROJECT/paulh -- don't copy it.)
REPO="$(cd "${LVL1}/../../../.." && pwd)"
S="${LVL1}/scripts"
GENERATED="${LVL1}/../data/generated_noblur"   # noblur set: metric reflects the FILL, not resolution sim
DATA="${LVL1}/outputs/data"
LOGDIR="${LVL1}/outputs/logs"
# 5 methods = every one with generated volumes present. NOTE: the intermediate ladder rungs
# (kmeans-only, kmeans+label_remap) have NO generated volumes, so a full 7-point
# texture-vs-Dice dose-response curve is NOT computable until those are generated.
METHODS="palette,v26_6_2_noisefill_v2,auglab_default,synthseg_em,synthseg_noem"
ERODE_ITERS=3
JOB_TIME="00:45:00"
mkdir -p "${DATA}" "${LOGDIR}"

source "${REPO}/datasets/open-ms/5_scripts_open-ms/00_utils/env.sh"
PY="${REPO}/.venv/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2   # thread contention, see run_texture_lvl1_openms.sh

run_eroded () {   # $1 = SOURCE (FLAIR/T1w), $2 = contrast label (flair/t1w)
    local src="$1" contrast="$2"
    local label="${contrast}_noblur_eroded"
    run_job --name "oms_ngf_erod_${contrast}" --gpus 4 --time "${JOB_TIME}" \
        --log "${LOGDIR}/ngf_${label}.log" --wait -- \
        bash -c "for R in 0 1 2 3; do \
            CUDA_VISIBLE_DEVICES=\$R ${PY} ${S}/compute_ngf_texture.py \
                --device cuda --rank \$R --world-size 4 --source ${src} --set-label ${label} \
                --erode-iters ${ERODE_ITERS} \
                --generated-root ${GENERATED} --methods ${METHODS} \
                --output-csv ${DATA}/ngf_${label}_rank\${R}.csv & \
        done; wait"
}

echo "== NGF (foreground-interior, erode x${ERODE_ITERS}) — FLAIR + T1w, 4 GPU shards each =="
run_eroded FLAIR flair
run_eroded T1w   t1w

echo "  shards written: $(ls ${DATA}/ngf_*_eroded_rank*.csv 2>/dev/null | wc -l)"
echo "== DONE — now re-plot with: ${PY} ${S}/plot_ngf_texture.py =="
