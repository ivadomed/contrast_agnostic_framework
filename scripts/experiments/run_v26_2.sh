#!/usr/bin/env bash
# Generate synthetic data for v26_2 across 4 GPU ranks.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
PY="${REPO_ROOT}/.venv/bin/python"

PIDS=()
for rank in 0 1 2 3; do
    run_job --name "gen_v26_2_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/gen${rank}.log" -- \
        "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
        --generator v26_2 --lhc --rank $rank --world-size 4 --device cuda:0 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
echo "Generation done!"
