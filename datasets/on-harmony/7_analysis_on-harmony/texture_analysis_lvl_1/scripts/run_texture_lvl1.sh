#!/usr/bin/env bash
# Level-1 texture analysis driver.
#
# Runs on the set_slot machine (4 GPUs). Steps:
#   0. metric self-test (must PASS before anything else)
#   1. compute metrics — 4 GPU shards in parallel (set_slot 0..3)
#   2. aggregate + plot (single process)
#
# Prereq: the 3360 generated volumes exist under data/generated/<method>/... .
# Usage:  bash run_texture_lvl1.sh
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # texture_analysis_lvl_1/
REPO="$(cd "${LVL1}/../../../.." && pwd)"                         # repo root
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
DATA="${LVL1}/outputs/data"
TABLES="${LVL1}/outputs/tables"
PLOTS="${LVL1}/outputs/plots"
DS_JSON="${REPO}/datasets/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/dataset.json"
mkdir -p "${DATA}" "${TABLES}" "${PLOTS}" /tmp/texlvl1

echo "== Step 0: metric self-test =="
"${PY}" "${S}/compute_texture_metrics.py" --sanity --device cuda

echo "== Step 1: compute metrics (4 GPU shards) =="
PIDS=()
for rank in 0 1 2 3; do
  set_slot "${rank}" "${PY}" "${S}/compute_texture_metrics.py" \
      --device cuda --rank "${rank}" --world-size 4 \
      --output-csv "${DATA}/metrics_rank${rank}.csv" \
      > "/tmp/texlvl1/compute_rank${rank}.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
echo "  shards done: $(ls ${DATA}/metrics_rank*.csv | wc -l)/4"

echo "== Step 2: aggregate + plot =="
"${PY}" "${S}/aggregate_texture_metrics.py" \
    --input "${DATA}/metrics_rank*.csv" --output-dir "${TABLES}"
"${PY}" "${S}/plot_texture_metrics.py" \
    --input "${DATA}/metrics_rank*.csv" --output-dir "${PLOTS}" --labels-json "${DS_JSON}"

echo "== DONE =="
echo "  tables → ${TABLES}"
echo "  plots  → ${PLOTS}"
