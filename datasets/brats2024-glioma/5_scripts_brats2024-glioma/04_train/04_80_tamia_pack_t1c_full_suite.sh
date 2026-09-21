#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the FULL BraTS T1c (contrast-enhanced T1) training suite:
# the 6-method comparison (baseline, auglab_default, synthseg_noEM, synthseg_EM, srcsm, OURS DualVal) + the 4
# causal-ablation ladder rungs (kmeans, +label_remap, +voronoi, v26_6_2-alone) = 10 wrappers x 3 folds (0 1 2).
# RUN_IDs + the pack layout come from 00_utils/t1c_runs.sh (single source of truth). Run ON tamia.
#
# run_job_pack_submit.sh launches EVERY recorded fold of a PACK_DIR concurrently, so 30 folds are split into three
# packs (own PACK_DIR + chain each, able to run on separate nodes):
#   A (12 folds) baseline, auglab_default, synthseg_EM, synthseg_noEM     B (12) OURS DualVal, rung5, rung2, rung3
#   C (6)  rung4 voronoi (3 folds share GPU 3) + srcsm (one GPU per fold — PACK_GPU_MAP "3 3 3 0 1 2")
#
# Guards learned on the T2f port: after RECORDING (which bakes absolute paths into the cmd files) every cmd file is
# checked to target dataset 054 / the t1c tree on /scratch and to contain NO /project results path, and row counts are
# checked — BEFORE anything is submitted. Chain job ids are saved to <PACKROOT>/t1c_launch_<TS>.txt so predict/eval
# can be queued behind them with native Slurm dependencies.
#
# Usage (on tamia):  bash 04_80_tamia_pack_t1c_full_suite.sh          # PACK_CHAIN=10 to change chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
source "${HERE}/../00_utils/env_t1c.sh"
source "${ROOT}/scripts/cluster/tamia_env.sh"
source "${HERE}/../00_utils/t1c_runs.sh"

PACKROOT="/scratch/p/paulh/brats2024-glioma/_packruns"
declare -A PDIR=( [A]="${PACKROOT}/t1c_suiteA_${T1C_TS}" [B]="${PACKROOT}/t1c_suiteB_${T1C_TS}" [C]="${PACKROOT}/t1c_suiteC_${T1C_TS}" )
for p in A B C; do
  [ ! -e "${PDIR[$p]}/index.tsv" ] || { echo "ERROR: ${PDIR[$p]} already recorded — refusing to re-record (would duplicate rows)" >&2; exit 1; }
  mkdir -p "${PDIR[$p]}"
done

run_id_of() { local n="$1" row; for row in "${T1C_RUNS[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"; [ "${name}" = "$n" ] && { echo "${run}"; return; }; done; echo "no run for $n" >&2; return 1; }

declare -A NROWS=([A]=0 [B]=0 [C]=0)
for row in "${T1C_TRAIN[@]}"; do IFS='|' read -r name wrapper pack <<<"${row}"
  run="$(run_id_of "${name}")"
  echo "[t1c-pack] pack ${pack}: ${name} -> ${run}"
  RUN_JOB_PACK_DIR="${PDIR[$pack]}" bash "${HERE}/${wrapper}" "${run}"
  NROWS[$pack]=$(( ${NROWS[$pack]} + 3 ))
done

echo "[t1c-pack] verifying recorded commands (row counts, dataset 054, /scratch t1c results, no /project paths)"
for p in A B C; do
  n=$(wc -l < "${PDIR[$p]}/index.tsv")
  [ "${n}" = "${NROWS[$p]}" ] || { echo "ERROR: pack ${p} recorded ${n} rows, expected ${NROWS[$p]}" >&2; exit 1; }
  while IFS=$'\t' read -r cmdfile _rest; do
    [ -n "${cmdfile}" ] || continue
    grep -q "nnUNetv2_train 054 " "${cmdfile}" || { echo "ERROR: ${cmdfile} is not dataset 054" >&2; exit 1; }
    grep -q "/scratch/p/paulh/brats2024-glioma/8_results/01_predictions/brats2024_glioma_model/t1c/" "${cmdfile}" || { echo "ERROR: ${cmdfile} results path is not the scratch t1c tree" >&2; exit 1; }
    ! grep -q "nnUNet_results=[^ ]*/project/" "${cmdfile}" || { echo "ERROR: ${cmdfile} writes results under /project (quota!)" >&2; exit 1; }
  done < "${PDIR[$p]}/index.tsv"
  echo "  pack ${p}: ${n} rows OK"
done

LAUNCH="${PACKROOT}/t1c_launch_${T1C_TS}.txt"; : > "${LAUNCH}"
submit_pack() {  # $1 = pack letter, extra env passed through
  local out; out="$(PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME=23:59:00 PACK_CHAIN="${PACK_CHAIN:-10}" PACK_USE_MPS=0 \
      PACK_JOB_NAME="brats_t1c_suite$1" bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PDIR[$1]}")"
  echo "${out}"
  echo "PACK_$1=$(echo "${out}" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')" >> "${LAUNCH}"
}
submit_pack A
submit_pack B
PACK_GPU_MAP="3 3 3 0 1 2" submit_pack C
echo "[t1c-pack] launch record: ${LAUNCH}"; cat "${LAUNCH}"
