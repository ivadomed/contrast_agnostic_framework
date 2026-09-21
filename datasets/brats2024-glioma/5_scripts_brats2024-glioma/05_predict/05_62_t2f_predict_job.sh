#!/bin/bash
# Whole-node predict job body for the brats2024-glioma T2f/FLAIR runs. Submitted by
# 05_61_tamia_pack_predict_t2f.sh (which passes GROUP, PACK_DIR, HERE_DIR via --export and the
# node/time/dependency flags on the sbatch command line — the #SBATCH lines below are just defaults).
#
# Steps: (1) refuse to run unless every selected run has checkpoint_best + checkpoint_final for
# folds 0 1 2, (2) RECORD each run's predict wrapper into a pack dir (run_job pack-record mode —
# no compute), (3) launch all recorded fold-commands concurrently pinned round-robin over the 4
# GPUs, (4) AUDIT that every fold/contrast produced exactly the expected number of predictions.
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
set -uo pipefail
: "${GROUP:?}" "${PACK_DIR:?}" "${HERE_DIR:?}"
cd "${HERE_DIR}/05_predict"
source ../00_utils/env.sh
source ../../../../scripts/cluster/tamia_env.sh
source ../00_utils/t2f_runs.sh
export TRAINING_CONTRAST=t2f
RD_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/t2f"
echo "[predict-job] group=${GROUP} host=$(hostname) job=${SLURM_JOB_ID:-?} PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

runs=()
for row in "${T2F_RUNS[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"; [ "$g" = "${GROUP}" ] && runs+=("${row}"); done
[ ${#runs[@]} -gt 0 ] || { echo "[predict-job] ERROR: no runs for group ${GROUP}" >&2; exit 2; }

missing=0
for row in "${runs[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"
  for k in 0 1 2; do for ck in checkpoint_final checkpoint_best; do
    ls "${RD_BASE}/${cat}/${run}"/Dataset053_*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { echo "[predict-job] MISSING ${ck} ${name} fold${k}" >&2; missing=1; }
  done; done
done
[ "${missing}" = "0" ] || { echo "[predict-job] ERROR: training not complete — refusing to predict." >&2; exit 1; }
echo "[predict-job] all ${#runs[@]} run(s) have checkpoint_best+final for folds 0-2"

for row in "${runs[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"
  RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run}" || { echo "[predict-job] ERROR recording ${name}" >&2; exit 1; }
done
n_rec=$(wc -l < "${PACK_DIR}/index.tsv"); n_exp=$(( ${#runs[@]} * 3 ))
echo "[predict-job] recorded ${n_rec} fold-commands (expected ${n_exp})"
[ "${n_rec}" = "${n_exp}" ] || { echo "[predict-job] ERROR: recorded row count mismatch" >&2; exit 1; }
# every recorded command must target the T2f dataset (053) + t2f results tree — a stale/collided cmd would silently mis-predict
while IFS=$'\t' read -r cmdfile _rest; do
  [ -n "${cmdfile}" ] || continue
  grep -q -- "-d 053 " "${cmdfile}" && grep -q "/t2f/" "${cmdfile}" || { echo "[predict-job] ERROR: ${cmdfile} does not target dataset 053 / t2f" >&2; exit 1; }
done < "${PACK_DIR}/index.tsv"
echo "[predict-job] all ${n_rec} recorded commands target dataset 053 / t2f"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
declare -a PIDS NAMES; i=0
while IFS=$'\t' read -r cmdfile log name donefile; do
  [ -n "${cmdfile}" ] || continue
  gpu=$(( i % 4 )); plog="${cmdfile%.sh}.log"
  echo "[predict-job] launch '${name}' on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1 &
  PIDS[$i]=$!; NAMES[$i]="${name}"; i=$(( i + 1 )); sleep 5
done < "${PACK_DIR}/index.tsv"
echo "[predict-job] ${i} fold-commands launched; waiting..."
rc_all=0
for j in "${!PIDS[@]}"; do
  if wait "${PIDS[$j]}"; then echo "[predict-job] OK   '${NAMES[$j]}'"; else echo "[predict-job] FAIL '${NAMES[$j]}'"; rc_all=1; fi
done

echo "[predict-job] AUDIT: expecting ${T2F_N_TEST_CASES} predictions per fold/contrast"
audit_fail=0
for row in "${runs[@]}"; do IFS='|' read -r g name wrapper run cat msub <<<"${row}"
  for k in 0 1 2; do for c in ${T2F_CONTRASTS}; do
    n=$(ls "${RD_BASE}/${cat}/${run}/fold${k}/${c}"/*.nii.gz 2>/dev/null | wc -l)
    [ "${n}" = "${T2F_N_TEST_CASES}" ] || { echo "[predict-job] AUDIT FAIL ${name} fold${k} ${c}: ${n}/${T2F_N_TEST_CASES}"; audit_fail=1; }
  done; done
done
[ "${audit_fail}" = "0" ] && echo "[predict-job] AUDIT OK" || echo "[predict-job] AUDIT FAILED"
echo "[predict-job] done rc_all=${rc_all} audit_fail=${audit_fail}"
[ "${rc_all}" = "0" ] && [ "${audit_fail}" = "0" ]
