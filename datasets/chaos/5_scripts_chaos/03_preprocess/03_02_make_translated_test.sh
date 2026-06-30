#!/usr/bin/env bash
# Build translated copies of the chaos test sets (image+GT shifted left by --frac of
# the L axis) for the translation-robustness experiment. Small/fast; see the .py.
#   bash 03_02_make_translated_test.sh                 # default 50% left, both DS, 4 mods
#   bash 03_02_make_translated_test.sh --frac 0.25
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name chaos_make_translated_test --gpus 0 --cpus 4 --mem 16G --time 00:15:00 \
    --log /tmp/chaos_make_translated_test.log --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_02_make_translated_test.py" "$@"
