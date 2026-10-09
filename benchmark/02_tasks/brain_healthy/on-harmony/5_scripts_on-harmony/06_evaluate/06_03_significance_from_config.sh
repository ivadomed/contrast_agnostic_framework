#!/usr/bin/env bash
# Paired significance testing from a YAML config (same config as 06_02), using the SHARED
# config-driven significance tester (same as chaos/open-ms — no on-harmony-specific stats
# code). Reads <metrics_dir>/<category>_<run_id>/fold*/eval_all.csv for every run in the
# config (standard 02_metrics layout) and writes <metrics_dir>/<output_prefix>_significance.md.
# Run 06_01_evaluate_testset.sh (predict+eval) and 06_02_aggregate_from_config.sh first.
#
# ⚠️ On-harmony's significance test is currently anti-conservative for ALL modalities
# (per-image units from 4 test subjects, not per-patient) — see project memory
# project_onharmony_significance_rerun_20260921 and each config's own header comment.
#
# Usage:
#   bash 06_03_significance_from_config.sh configs/on-harmony_T1w.yaml
#   bash 06_03_significance_from_config.sh configs/on-harmony_T2w.yaml
#   bash 06_03_significance_from_config.sh configs/on-harmony_dwi_ap.yaml
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CFG="${1:?Usage: $0 <config.yaml>}"; shift
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }

.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/significance_from_config.py" "${CFG}" "$@"
