#!/usr/bin/env bash
# Aggregation + significance for the brats2024-glioma T1c (contrast-enhanced T1) training modality. Thin driver
# over the shared 00_commun_scripts layer — no logic of its own. Re-run it whenever the T1c runs
# change (e.g. once srcsm's predict+eval land: its row is blank until then).
#
#   1. per-modality table + heatmaps            (aggregate_from_config.py, configs/brats_t1c_01_results.yaml)
#   2. paired significance, Dice + HD95         (significance_from_config.py; HD95 goes to a separate
#                                                prefix so it cannot overwrite the Dice report)
#   3. 4-modality combined table (t1n+t2w+t2f+t1c)  (combined_modality_summary.py; its OWN output_dir,
#                                                the 2-modality headline is untouched)
#   4. causal-ablation ladder                   (06_33_ladder_summary_t1c.py -> shared ladder engine)
#
# Runs itself under run_job (CPU) — aggregation/bootstrap is not login-node work.
# Usage: bash 06_32_aggregate_t1c.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

if [ -z "${_T1C_AGG_IN_JOB:-}" ]; then
    LOGDIR="${METRICS_ROOT}/brats2024_glioma_model/t1c/_logs"; mkdir -p "${LOGDIR}"
    run_job --name brats_t1c_aggregate --gpus 0 --cpus 4 --mem 16G --time 1:00:00 \
        --log "${LOGDIR}/aggregate_$(date +%Y%m%d_%H%M%S).log" --wait -- \
        env _T1C_AGG_IN_JOB=1 bash "${HERE}/$(basename "${BASH_SOURCE[0]}")"
    exit $?
fi

PY="${PROJECT_ROOT}/.venv/bin/python"
COMMON="${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate"
CFG_T1C="${HERE}/configs/brats_t1c_01_results.yaml"
CFG_4MOD="${HERE}/configs/brats_combined_4mod_01_results.yaml"

echo "=== [1/4] per-modality aggregation (t1c) ==="
"${PY}" "${COMMON}/aggregate_from_config.py" "${CFG_T1C}"

echo "=== [2/4] significance: Dice, then HD95 (own prefix) ==="
"${PY}" "${COMMON}/significance_from_config.py" "${CFG_T1C}" --metric dice
HD95_CFG="$(mktemp --suffix=.yaml -p "${SCRATCH:?SCRATCH must be set}")"
sed 's/^output_prefix:.*/output_prefix: "03_01_results_hd95"/' "${CFG_T1C}" > "${HD95_CFG}"
"${PY}" "${COMMON}/significance_from_config.py" "${HD95_CFG}" --metric hd95
rm -f "${HD95_CFG}"

echo "=== [3/4] combined 4-modality table (t1n + t2w + t2f + t1c) ==="
"${PY}" "${COMMON}/combined_modality_summary.py" "${CFG_4MOD}"

echo "=== [4/4] causal-ablation ladder (t1c) ==="
"${PY}" "${HERE}/06_33_ladder_summary_t1c.py"
echo "[$(date '+%H:%M:%S')] done"
