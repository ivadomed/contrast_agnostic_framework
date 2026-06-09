#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Run all 4 training methods sequentially (baseline → V26_6 → SynthSeg-A → SynthSeg-B).
# Each method uses all 4 GPUs (one fold per GPU) in parallel.
# Sequential across methods is faster than parallel (avoids GPU memory contention).
#
# Usage: bash scripts/nnunet_onharmony/run_all_training.sh [--start-from <method>]
#   --start-from: skip to a specific method (baseline|v26_6|synthseg_a|synthseg_b)
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

START_FROM="${1:-baseline}"
[[ "$1" == "--start-from" ]] && START_FROM="${2:-baseline}" || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_method() {
    local method="$1" script="$2"
    log "=== Starting ${method} ==="
    bash "$script"
    log "=== ${method} complete ==="
}

# Methods in order
declare -a METHODS=("baseline" "v26_6" "synthseg_a" "synthseg_b")
declare -A SCRIPTS=(
    ["baseline"]="scripts/nnunet_onharmony/03_train_baseline.sh"
    ["v26_6"]="scripts/nnunet_onharmony/03_train_v26_6.sh"
    ["synthseg_a"]="scripts/nnunet_onharmony/03_train_synthseg_a.sh"
    ["synthseg_b"]="scripts/nnunet_onharmony/03_train_synthseg_b.sh"
)

# Find starting index
START_IDX=0
for i in "${!METHODS[@]}"; do
    [[ "${METHODS[$i]}" == "$START_FROM" ]] && START_IDX=$i && break
done

log "Starting from method: ${METHODS[$START_IDX]}"

for i in $(seq $START_IDX $((${#METHODS[@]}-1))); do
    method="${METHODS[$i]}"
    run_method "$method" "${SCRIPTS[$method]}"
done

log "All training complete."
