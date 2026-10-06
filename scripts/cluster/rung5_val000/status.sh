#!/usr/bin/env bash
# Progress + checkpoint-selection audit of the val000 real-fill runs (light: reads text files only).
# Per fold: last epoch in training_log_*.txt, checkpoint_final present, and debug.json's
# validation_uses_augmentation (must be False: checkpoint_best chosen on REAL validation images).
#   bash scripts/cluster/rung5_val000/status.sh [--problems]
cd /project/aip-jcohen/paulh/mri_synthesis_project
RUNS="${SCRATCH:?}/rung5_val000/run_ids.txt"
[ -f "${RUNS}" ] || { echo "no ${RUNS}: run vulcan_launch.sh --record first" >&2; exit 1; }
while read -r rid; do
  d=$(ls -d benchmark/02_tasks/*/*/8_results_*/01_predictions/*/*/*/"${rid}" 2>/dev/null | head -1)
  [ -n "$d" ] || { echo "MISSING-DIR ${rid}"; continue; }
  for k in 0 1 2; do
    f=$(ls -d "$d"/Dataset*/*/fold_${k} 2>/dev/null | head -1)
    ep=$(grep -hoE "Epoch [0-9]+" "$f"/training_log_*.txt 2>/dev/null | tail -1 | awk '{print $2}')
    va=$(grep -hoE '"validation_uses_augmentation": *"?[A-Za-z]+' "$f/debug.json" 2>/dev/null | grep -oE '[A-Za-z]+$')
    fin=$([ -f "$f/checkpoint_final.pth" ] && echo final || echo -)
    flag=""; [ "${va}" = True ] && flag=" <-- VAL100"; [ -z "${ep}" ] && flag="${flag} <-- NOT STARTED"
    [ "${1:-}" = --problems ] && [ -z "${flag}" ] && continue
    printf "%-62s fold%s epoch=%-5s val_aug=%-6s %s%s\n" "${rid}" "$k" "${ep:--}" "${va:--}" "${fin}" "${flag}"
  done
done < "${RUNS}"
