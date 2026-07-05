#!/usr/bin/env bash
# Level-1 texture analysis driver (census + NMI), for BOTH the with-blur (training-config) set
# and the no-blur ablation set.
#
# Runs on the set_slot machine (4 GPUs). Steps:
#   0. metric self-test (must PASS)
#   1. compute metrics — blur set + noblur set, each 4 GPU shards; controls once (set=ref)
#   2. aggregate + plot over the combined CSVs (blur vs noblur side by side)
#
# Prereq: data/generated/  (with-blur) and data/generated_noblur/ (ablation) both populated.
# Usage:  bash run_texture_lvl1.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
DATA="${LVL1}/outputs/data"
TABLES="${LVL1}/outputs/tables"
PLOTS="${LVL1}/outputs/plots"
DS_JSON="${REPO}/datasets/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/dataset.json"
METHODS="palette,synthseg_em,synthseg_noem,auglab_default"
mkdir -p "${DATA}" "${TABLES}" "${PLOTS}" /tmp/texlvl1

echo "== Step 0: metric self-test =="
"${PY}" "${S}/compute_texture_metrics.py" --sanity --device cuda

run_set () {   # $1 = set label, $2 = generated root
  local label="$1" root="$2"
  [ -d "${root}" ] || { echo "  skip ${label}: ${root} missing"; return; }
  echo "== compute ${label} (4 GPU shards) =="
  local PIDS=()
  for rank in 0 1 2 3; do
    set_slot "${rank}" "${PY}" "${S}/compute_texture_metrics.py" \
        --device cuda --rank "${rank}" --world-size 4 \
        --generated-root "${root}" --set-label "${label}" --methods "${METHODS}" \
        --output-csv "${DATA}/metrics_${label}_rank${rank}.csv" \
        > "/tmp/texlvl1/compute_${label}_rank${rank}.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
}

echo "== Step 1: compute metrics =="
run_set blur   "${LVL1}/data/generated"
run_set noblur "${LVL1}/data/generated_noblur"
# inline controls (gamma/histeq) once — set=ref, source-only
set_slot 0 "${PY}" "${S}/compute_texture_metrics.py" --device cuda \
    --methods "gamma,histeq" --set-label ref \
    --output-csv "${DATA}/metrics_ref_rank0.csv" > /tmp/texlvl1/compute_ref.log 2>&1
echo "  shards: $(ls ${DATA}/metrics_*_rank*.csv 2>/dev/null | wc -l)"

echo "== Step 2: aggregate + plot (combined) =="
"${PY}" "${S}/aggregate_texture_metrics.py" \
    --input "${DATA}/metrics_*_rank*.csv" --output-dir "${TABLES}"
"${PY}" "${S}/plot_texture_metrics.py" \
    --input "${DATA}/metrics_*_rank*.csv" --output-dir "${PLOTS}" --labels-json "${DS_JSON}"

echo "== DONE =="
echo "  tables → ${TABLES}   plots → ${PLOTS}"
