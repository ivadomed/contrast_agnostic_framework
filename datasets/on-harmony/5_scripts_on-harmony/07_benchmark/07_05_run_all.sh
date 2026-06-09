#!/usr/bin/env bash
# =============================================================================
# Full benchmark pipeline: train → predict → evaluate.
# Queue multiple methods sequentially, with all folds of each method parallel.
# =============================================================================
# Usage:
#   bash scripts/benchmark/run_benchmark.sh [OPTIONS]
#
# Options:
#   METHODS="v26_6 synthseg_a"   methods to run (space-separated, default = all)
#   GPUS=0-3                     GPU slots
#   LABEL_SET=7class             label set
#   N_EPOCHS=500                epochs per method
#   SKIP_TRAIN=0                 1 = skip training (use existing RUN_IDs below)
#   SKIP_PREDICT=0               1 = skip prediction
#   SKIP_EVALUATE=0              1 = skip evaluation
#
# To run a subset:
#   METHODS="baseline synthseg_a" bash scripts/benchmark/run_benchmark.sh
#   SKIP_TRAIN=1 bash scripts/benchmark/run_benchmark.sh   # eval only

set -euo pipefail
SCRIPT_DIR="$(dirname "$0")"

METHODS="${METHODS:-baseline v26_6 synthseg_a synthseg_b}"
GPUS="${GPUS:-0-3}"
LABEL_SET="${LABEL_SET:-7class}"
N_EPOCHS="${N_EPOCHS:-500}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"
SKIP_EVALUATE="${SKIP_EVALUATE:-0}"

TRAINED_RUNS=()

for METHOD in $METHODS; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  METHOD=$METHOD  LABEL_SET=$LABEL_SET  GPUS=$GPUS"
    echo "════════════════════════════════════════════════════════"

    if [ "$SKIP_TRAIN" -eq 0 ]; then
        METHOD=$METHOD GPUS=$GPUS LABEL_SET=$LABEL_SET N_EPOCHS=$N_EPOCHS \
            bash "$SCRIPT_DIR/train.sh"
        # Capture the RUN_ID printed by train.sh
        RUN_ID=$(METHOD=$METHOD GPUS=$GPUS LABEL_SET=$LABEL_SET \
            bash -c 'source '"$SCRIPT_DIR"'/07_01_config.sh 2>/dev/null; echo "${METHOD}_$(date +%Y%m%d)"' 2>/dev/null || true)
    fi

    # Find the most recent run for this method if not tracking
    RUN_ID=$(ls -d "${NNUNET_RES}/${METHOD}_"* 2>/dev/null | tail -1 | xargs basename 2>/dev/null || true)
    [ -z "$RUN_ID" ] && echo "  No run found for $METHOD, skipping predict/eval" && continue

    echo "[pipeline] Using RUN_ID=$RUN_ID"
    TRAINED_RUNS+=("$RUN_ID")

    if [ "$SKIP_PREDICT" -eq 0 ]; then
        METHOD=$METHOD GPUS=$GPUS LABEL_SET=$LABEL_SET RUN_ID=$RUN_ID \
            bash "$SCRIPT_DIR/predict.sh"
    fi
done

if [ "$SKIP_EVALUATE" -eq 0 ] && [ ${#TRAINED_RUNS[@]} -gt 0 ]; then
    ALL_RUNS="${TRAINED_RUNS[*]}"
    echo ""
    echo "════ EVALUATION ════"
    RUN_ID="$ALL_RUNS" GPUS=$GPUS bash "$SCRIPT_DIR/evaluate.sh"
fi

echo ""
echo "════ BENCHMARK COMPLETE ════"
