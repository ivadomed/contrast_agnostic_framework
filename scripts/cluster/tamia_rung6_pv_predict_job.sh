#!/bin/bash
# Whole-node PREDICT job body for ONE rung-6 (PALETTE + boundary PV) pack. Submitted by
# `tamia_pack_rung6_pv.sh queue-post` with PACK + OUT exported (dependency/time on the sbatch line).
# Mirrors scripts/cluster/tamia_rung5_predict_job.sh (the rung-5 retrain's job): (1) refuse unless every
# run of the pack has checkpoint_best + checkpoint_final for folds 0-2, (2) RECORD each row's own predict
# wrapper into its own pack dir, (3) run all fold-commands pinned round-robin over the 4 GPUs,
# (4) AUDIT per-item prediction counts. Run table: scripts/cluster/tamia_rung6_pv_runs.sh.
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
set -uo pipefail
: "${PACK:?}" "${OUT:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
source scripts/cluster/tamia_rung6_pv_runs.sh
T=benchmark/02_tasks
echo "[predict-job] pack=${PACK} host=$(hostname) job=${SLURM_JOB_ID:-?} OUT=${OUT}"

envsetup() {   # <dir> <envfile> <tamia env> <tc>  (call inside a subshell)
    export TRAINING_CONTRAST="$4"; source "$T/$1/00_utils/$2"; source "scripts/cluster/$3"; export TRAINING_CONTRAST="$4"
}
rows=(); for r in "${RUNG6_ROWS[@]}"; do IFS='|' read -r name pack _ <<<"$r"; [ "$pack" = "$PACK" ] && rows+=("$r"); done
[ ${#rows[@]} -gt 0 ] || { echo "[predict-job] ERROR: no rows for pack ${PACK}" >&2; exit 2; }

# (1) checkpoints (the flipped mandible arms reuse the toothfairy2 model -- checked by its own row)
missing=0
for r in "${rows[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$r"
  case "$name" in *_sif) continue ;; esac
  ( envsetup "$dir" "$envf" "$tenv" "$tc"
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/nnUNet/${run}"
    for k in 0 1 2; do for ck in checkpoint_final checkpoint_best; do
      ls "${base}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { echo "[predict-job] MISSING ${ck} ${name} fold${k} (${base})" >&2; exit 1; }
    done; done ) || missing=1
done
[ "${missing}" = 0 ] || { echo "[predict-job] ERROR: training not complete -- refusing to predict." >&2; exit 1; }
echo "[predict-job] all ${#rows[@]} rows: checkpoint_best+final present for folds 0-2"

# (2) record
for r in "${rows[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$r"
  PD="${OUT}/predict_${name}"; mkdir -p "${PD}"
  # shellcheck disable=SC2086
  ( envsetup "$dir" "$envf" "$tenv" "$tc"; rung6_predict_extra_env "$name"
    RUN_JOB_PACK_DIR="${PD}" bash "$T/$dir/05_predict/${wrapper}" "${run}" ${args} ) \
    || { echo "[predict-job] ERROR recording ${name}" >&2; exit 1; }
  n=$(wc -l < "${PD}/index.tsv"); echo "[predict-job] ${name}: recorded ${n} fold-commands (expect 3)"
  [ "${n}" = 3 ] || exit 1
  while IFS=$'\t' read -r cmdfile _r; do
    grep -q -- "${run}" "${cmdfile}" && grep -q "/scratch/" "${cmdfile}" \
      || { echo "[predict-job] ERROR: ${cmdfile} does not reference ${run} on /scratch" >&2; exit 1; }
    case "$name" in *_sif) grep -q "_sif" "${cmdfile}" || { echo "[predict-job] ERROR: ${cmdfile} not on the flipped (_sif) inputs" >&2; exit 1; } ;; esac
  done < "${PD}/index.tsv"
done

# (3) launch all, pinned round-robin
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
declare -a PIDS NAMES; i=0
for r in "${rows[@]}"; do IFS='|' read -r name _ <<<"$r"
  while IFS=$'\t' read -r cmdfile log nm donefile; do
    [ -n "${cmdfile}" ] || continue
    gpu=$(( i % 4 )); plog="${cmdfile%.sh}.log"
    echo "[predict-job] launch '${nm}' on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1 &
    PIDS[$i]=$!; NAMES[$i]="${nm}"; i=$(( i + 1 )); sleep 5
  done < "${OUT}/predict_${name}/index.tsv"
done
echo "[predict-job] ${i} fold-commands launched; waiting..."
rc_all=0
for j in "${!PIDS[@]}"; do
  if wait "${PIDS[$j]}"; then echo "[predict-job] OK   '${NAMES[$j]}'"; else echo "[predict-job] FAIL '${NAMES[$j]}'"; rc_all=1; fi
done

# (3b) on-harmony ONLY: its predict wrapper (05_01_predict_common.sh) resamples the RAS-space predictions back
# to native geometry right after the shared driver returns. In pack-record mode that line runs BEFORE any
# prediction exists, so replay it here, after the packed predictions, with the dataset's own 05_02 script.
for r in "${rows[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$r"
  [ "${ekind}" = onh ] || continue
  ( envsetup "$dir" "$envf" "$tenv" "$tc"
    for k in 0 1 2; do for it in ${pexp}; do item="${it%:*}"
      .venv/bin/python "$T/$dir/05_predict/05_02_resample_predictions_to_native.py" \
        "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/nnUNet/${run}/fold${k}/final/${item}" \
        "${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set/${item}/images_native" || exit 1
    done; done ) || { echo "[predict-job] ERROR: resample-to-native failed for ${name}"; rc_all=1; }
done

# (4) audit (checkpoint_final predictions land under fold{k}/final/ -- on-harmony's default)
audit_fail=0
for r in "${rows[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$r"
  ( envsetup "$dir" "$envf" "$tenv" "$tc"
    # cross-dataset (mandible) envs name the source model's type TF2_MODEL_TYPE
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE:-${TF2_MODEL_TYPE}}/${tc}/nnUNet/${run}"
    bad=0
    for k in 0 1 2; do for it in ${pexp}; do item="${it%:*}"; want="${it##*:}"
      n=$(ls "${base}"/fold${k}/${item}/*.nii.gz "${base}"/fold${k}/final/${item}/*.nii.gz 2>/dev/null | wc -l)
      [ "${n}" = "${want}" ] || { echo "[predict-job] AUDIT FAIL ${name} fold${k} ${item}: ${n}/${want} (${base})"; bad=1; }
    done; done
    exit ${bad} ) || audit_fail=1
done
[ "${audit_fail}" = 0 ] && echo "[predict-job] AUDIT OK" || echo "[predict-job] AUDIT FAILED"
echo "[predict-job] done rc_all=${rc_all} audit_fail=${audit_fail}"
[ "${rc_all}" = 0 ] && [ "${audit_fail}" = 0 ]
