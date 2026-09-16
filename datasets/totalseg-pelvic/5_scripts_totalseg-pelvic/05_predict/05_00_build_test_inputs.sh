#!/usr/bin/env bash
# Materialize both modalities' held-out test sets as nnU-Net predict inputs. Run once
# after 02_nnunet/02_00_convert.py, before any predict wrapper.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/05_predict/05_00_build_test_inputs.py
