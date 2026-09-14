#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
.venv/bin/python "$(dirname "$0")/05_00b_build_psma_test_inputs.py"
