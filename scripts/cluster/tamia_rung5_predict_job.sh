#!/bin/bash
# Whole-node PREDICT job body for the rung-5 retrains (chaos t2spir, on-harmony T1w, brats t2w).
# Submitted by tamia_pack_rung5_retrain.sh queue-post (ROOT exported; dependency/time on the sbatch line).
# (1) refuse unless every run has checkpoint_best + checkpoint_final for folds 0-2, (2) RECORD each run's
# predict wrapper into its OWN dataset-qualified pack dir (never share one -- see the project notes PACK_DIR rule),
# (3) run the 9 fold-commands pinned round-robin over the 4 GPUs, (4) AUDIT per-item prediction counts against
# the OLD rung-5 runs' counts (measured 2026-10-03 on Vulcan).
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
set -uo pipefail
: "${ROOT:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
source "${ROOT}/RUN_IDS.env"
T=benchmark/02_tasks

# name | dataset scripts dir | env file | tamia env | TRAINING_CONTRAST | predict wrapper | RUN_ID | category dir (model/contrast) | expected "item:n ..."
ROWS=(
"brats|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|env_t2w.sh|tamia_env.sh|t2w|05_predict/05_17_predict_t2w_v26_6_2_train050_val100.sh|${BRATS_RUN}|t1n:70 t1c:70 t2w:70 t2f:70"
"onharmony|$T/brain_healthy/on-harmony/5_scripts_on-harmony|env.sh|tamia_env_onharmony.sh|T1w|05_predict/05_21_predict_t1w_v26_6_2_train050_val100.sh|${ONH_RUN}|T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:8"
"chaos|$T/abdomen_healthy/chaos/5_scripts_chaos|env_t2spir.sh|tamia_env_chaos.sh|t2spir|05_predict/05_22_predict_t2spir_v26_6_2_train050_val100.sh|${CHAOS_RUN}|t1in:4 t1out:4 t2spir:4 ct:20"
)
# SKIP (optional, space-separated dataset names: brats onharmony chaos): leave those out (e.g. already done separately)
_F=(); for row in "${ROWS[@]}"; do [[ " ${SKIP:-} " == *" ${row%%|*} "* ]] || _F+=("$row"); done; ROWS=("${_F[@]}")
[ ${#ROWS[@]} -gt 0 ] || { echo "[predict-job] nothing to do (SKIP='${SKIP:-}')"; exit 0; }
echo "[predict-job] host=$(hostname) job=${SLURM_JOB_ID:-?} ROOT=${ROOT} datasets: $(for r in "${ROWS[@]}"; do printf '%s ' "${r%%|*}"; done)"

# envsetup <dir> <envfile> <tamia env> <tc>: source in the CURRENT shell (call inside a subshell)
envsetup() { export TRAINING_CONTRAST="$4"; source "$1/00_utils/$2"; source "scripts/cluster/$3"; export TRAINING_CONTRAST="$4"; }

# (1) checkpoints
missing=0
for row in "${ROWS[@]}"; do IFS='|' read -r name dir envf tenv tc wrapper run exp <<<"${row}"
  ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}"
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/nnUNet/${run}"
    for k in 0 1 2; do for ck in checkpoint_final checkpoint_best; do
      ls "${base}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { echo "[predict-job] MISSING ${ck} ${name} fold${k} (${base})" >&2; exit 1; }
    done; done ) || missing=1
done
[ "${missing}" = 0 ] || { echo "[predict-job] ERROR: training not complete -- refusing to predict." >&2; exit 1; }
echo "[predict-job] all 3 runs have checkpoint_best+final for folds 0-2"

# (2) record
for row in "${ROWS[@]}"; do IFS='|' read -r name dir envf tenv tc wrapper run exp <<<"${row}"
  PD="${ROOT}/predict_${name}_${tc}"; mkdir -p "${PD}"
  ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}"
    RUN_JOB_PACK_DIR="${PD}" bash "${dir}/${wrapper}" "${run}" ) || { echo "[predict-job] ERROR recording ${name}" >&2; exit 1; }
  n=$(wc -l < "${PD}/index.tsv"); echo "[predict-job] ${name}: recorded ${n} fold-commands (expect 3)"
  [ "${n}" = 3 ] || exit 1
  while IFS=$'\t' read -r cmdfile _r; do
    grep -q -- "${run}" "${cmdfile}" && grep -q "/scratch/" "${cmdfile}" || { echo "[predict-job] ERROR: ${cmdfile} does not reference ${run} on /scratch" >&2; exit 1; }
  done < "${PD}/index.tsv"
done

# (3) launch all 9, pinned round-robin
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
declare -a PIDS NAMES; i=0
for row in "${ROWS[@]}"; do IFS='|' read -r name dir envf tenv tc wrapper run exp <<<"${row}"
  while IFS=$'\t' read -r cmdfile log nm donefile; do
    [ -n "${cmdfile}" ] || continue
    gpu=$(( i % 4 )); plog="${cmdfile%.sh}.log"
    echo "[predict-job] launch '${nm}' on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1 &
    PIDS[$i]=$!; NAMES[$i]="${nm}"; i=$(( i + 1 )); sleep 5
  done < "${ROOT}/predict_${name}_${tc}/index.tsv"
done
echo "[predict-job] ${i} fold-commands launched; waiting..."
rc_all=0
for j in "${!PIDS[@]}"; do
  if wait "${PIDS[$j]}"; then echo "[predict-job] OK   '${NAMES[$j]}'"; else echo "[predict-job] FAIL '${NAMES[$j]}'"; rc_all=1; fi
done

# (3b) on-harmony ONLY: its predict INPUT is RAS-canonical, so predictions come out RAS and must be resampled back to native
# geometry (05_02) before scoring against the native-space GT. The shim does this right after the driver returns -- but in
# pack-RECORD mode that is at record time, before any prediction exists (silent no-op), so the x-mirrored RAS predictions were
# scored against LAS GT (2026-10-05: OOD Dice 58 -> 32, caught by comparing against the old run). Do it here, after the folds ran.
for row in "${ROWS[@]}"; do IFS='|' read -r name dir envf tenv tc wrapper run exp <<<"${row}"
  [ "${name}" = onharmony ] || continue
  ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}"
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/nnUNet/${run}"
    for k in 0 1 2; do for it in ${exp}; do item="${it%%:*}"
      .venv/bin/python "${dir}/05_predict/05_02_resample_predictions_to_native.py" "${base}/fold${k}/final/${item}" "${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set/${item}/images_native"
    done; done ) || { echo "[predict-job] ERROR resampling on-harmony predictions to native" >&2; rc_all=1; }
done

# (4) audit
audit_fail=0
for row in "${ROWS[@]}"; do IFS='|' read -r name dir envf tenv tc wrapper run exp <<<"${row}"
  ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}"
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/nnUNet/${run}"
    bad=0
    for k in 0 1 2; do for it in ${exp}; do item="${it%%:*}"; want="${it##*:}"
      n=$(ls "${base}"/fold${k}/${item}/*.nii.gz "${base}"/fold${k}/final/${item}/*.nii.gz 2>/dev/null | wc -l)
      [ "${n}" = "${want}" ] || { echo "[predict-job] AUDIT FAIL ${name} fold${k} ${item}: ${n}/${want}"; bad=1; }
    done; done
    exit ${bad} ) || audit_fail=1
done
[ "${audit_fail}" = 0 ] && echo "[predict-job] AUDIT OK" || echo "[predict-job] AUDIT FAILED"
echo "[predict-job] done rc_all=${rc_all} audit_fail=${audit_fail}"
[ "${rc_all}" = 0 ] && [ "${audit_fail}" = 0 ]
