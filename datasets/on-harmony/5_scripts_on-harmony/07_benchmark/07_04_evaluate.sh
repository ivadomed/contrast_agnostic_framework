#!/usr/bin/env bash
# =============================================================================
# Ensemble fold predictions, resample to native space, compute Dice.
# Then print comparison table across all evaluated run IDs.
# =============================================================================
# Usage:
#   bash scripts/benchmark/evaluate.sh [OPTIONS]
#
# Options (env vars):
#   RUN_ID=v26_6_20260601_xxxxxx     REQUIRED: run to evaluate (can be multiple, space-sep)
#   GPUS=0-3                         CPU slots for parallel processing
#   CONTRASTS="T1w T2w ..."          contrasts to evaluate (default = all 6)
#   FORCE=0                          recompute even if dice.json exists
#
# Examples:
#   RUN_ID=synthseg_a_20260601_154222 bash scripts/benchmark/evaluate.sh
#   RUN_ID="synthseg_a_20260601_154222 synthseg_b_20260601_234120" bash scripts/benchmark/evaluate.sh
set -euo pipefail
source "$(dirname "$0")/07_01_config.sh"

if [ -z "${RUN_ID:-}" ] || [ "$RUN_ID" = "auto" ]; then
    echo "[evaluate] ERROR: RUN_ID must be set" >&2; exit 1
fi

cd "$PROJECT_ROOT"
CONTRASTS="${CONTRASTS:-T1w T2w bold dwi_ap epi_ap gre_echo1_mag}"
FORCE="${FORCE:-0}"
GPU_ARR=($GPU_LIST)
N_GPUS=${#GPU_ARR[@]}

# Split contrasts into N_GPUS chunks for parallel ensemble+dice
CONTRAST_ARR=($CONTRASTS)
N_CONTRASTS=${#CONTRAST_ARR[@]}
CHUNK_SIZE=$(( (N_CONTRASTS + N_GPUS - 1) / N_GPUS ))

for RUN in $RUN_ID; do
    echo "[evaluate] === $RUN ==="
    declare -A PIDS
    for ((i=0; i<N_GPUS; i++)); do
        START=$((i * CHUNK_SIZE))
        END=$(( START + CHUNK_SIZE ))
        [ $END -gt $N_CONTRASTS ] && END=$N_CONTRASTS
        [ $START -ge $N_CONTRASTS ] && break

        CHUNK="${CONTRAST_ARR[@]:$START:$((END - START))}"
        SLOT="${GPU_ARR[$i]}"

        run_job --name "benchmark_eval_${RUN}_chunk${i}" \
            --gpus 0 --slot "${SLOT}" --wait \
            --log "/tmp/eval_${RUN}_chunk${i}.log" -- \
            $PY "$PROJECT_ROOT/scripts/nnunet_onharmony/ensemble_and_dice.py" \
                "$RUN" $CHUNK &
        PIDS[$i]=$!
    done
    for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}"; done

    # Print per-contrast results for this run
    for CONTRAST in $CONTRASTS; do
        DICE_JSON="$EVAL_DIR/$RUN/$CONTRAST/dice.json"
        if [ -f "$DICE_JSON" ]; then
            MEAN=$($PY -c "
import json, numpy as np
d = json.load(open('$DICE_JSON'))['dice']
print(f'{np.nanmean(list(d.values())):.3f}')
" 2>/dev/null)
            echo "  $CONTRAST: $MEAN"
        fi
    done
done

# Print full comparison table (all runs in eval/ that have dice.json)
echo ""
$PY "$PROJECT_ROOT/scripts/nnunet_onharmony/ensemble_and_dice.py" --summary
