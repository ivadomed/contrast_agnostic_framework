#!/usr/bin/env bash
# Rung 4.5 on-harmony tail (trained on TamIA, tamia_pack_onharmony.sh): run ON VULCAN once the TamIA pack is done.
#   (1) for each TamIA RUN_ID: refuse unless checkpoint_best + checkpoint_final exist for folds 0-2 on TamIA;
#   (2) rsync the run dir to the SAME relative place the rung-4 lblvor runs sit on Vulcan
#       (8_results_on-harmony/01_predictions/on_harmony_model/<tc>/auglab/<RID>);
#   (3) queue scripts/cluster/rung45_flat/post_run.sh <RID> as a CPU run_job (predict on Vulcan with checkpoint_best,
#       RAS->native resample inside the predict job, evaluate into .../ablations, audit vs the old rung 4).
# queue_post.sh cannot do this: it keys on fold jobs in Vulcan's sacct, and these folds ran on TamIA.
#   bash scripts/cluster/rung45_flat/onharmony_fetch_and_post.sh [--dry-run]
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
IDS="${SCRATCH:?}/rung45_flat/run_ids_tamia.txt"
TS=/scratch/p/paulh/on-harmony/8_results/01_predictions/on_harmony_model
VS=benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model
source scripts/job_runner/run_job.sh
export RUN_JOB_ACCOUNT=aip-jcohen
while read -r rid; do
  [ -n "${rid}" ] || continue
  tc="$(sed -E 's/^on-harmony_(T1w|T2w|dwi_ap)_.*/\1/' <<<"${rid}")"
  src="${TS}/${tc}/auglab/${rid}"; dst="${VS}/${tc}/auglab/${rid}"
  n=$(ssh tamia.alliancecan.ca "ls ${src}/Dataset*/*/fold_{0,1,2}/checkpoint_{best,final}.pth 2>/dev/null | wc -l")
  [ "$n" = 6 ] || { echo "[r45-onh] ${rid}: ${n}/6 checkpoints on TamIA -- training not finished, skip"; continue; }
  echo "[r45-onh] ${rid}: checkpoints complete -> ${dst}"
  [ "${DRY}" = 1 ] && continue
  mkdir -p "${dst}"
  rsync -a "tamia.alliancecan.ca:${src}/" "${dst}/"
  m=$(ls "${dst}"/Dataset*/*/fold_{0,1,2}/checkpoint_{best,final}.pth 2>/dev/null | wc -l)
  [ "$m" = 6 ] || { echo "[r45-onh] ${rid}: only ${m}/6 checkpoints after rsync -- NOT queueing"; continue; }
  name="r45post_${rid}"; name="${name:0:120}"
  squeue -u "$USER" -h -o '%j' | grep -qxF "${name}" && { echo "[r45-onh] ${rid}: controller already queued"; continue; }
  run_job --name "${name}" --gpus 0 --cpus 8 --mem 64G --time 16:00:00 \
      --log "${SCRATCH}/rung45_flat/post/${rid}.job.log" -- bash scripts/cluster/rung45_flat/post_run.sh "${rid}"
  sleep 2
done < "${IDS}"
