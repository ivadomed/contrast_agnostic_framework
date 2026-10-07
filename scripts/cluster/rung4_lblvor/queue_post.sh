#!/usr/bin/env bash
# Queue one post-training controller (post_run.sh: verify -> predict -> evaluate -> audit) PER new rung-4 lblvor run, each
# behind `afterany:` its own 3 fold jobs (latest submission of each fold, so resubmitted folds count). Idempotent: skips
# runs whose controller is queued/running or whose status ends in DONE.
#   bash scripts/cluster/rung4_lblvor/queue_post.sh [--dry-run] [RUN_ID_REGEX]
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
PAT="${1:-.}"
OUTD="${SCRATCH:?}/rung4_lblvor/post"; mkdir -p "${OUTD}"
RUNS="${SCRATCH}/rung4_lblvor/run_ids.txt"
source scripts/job_runner/run_job.sh
export RUN_JOB_ACCOUNT=aip-jcohen
SACCT="$(sacct -u "$USER" -S 2026-10-07T16:00 -X -n -P -o JobID,JobName,State)"
QUEUED="$(squeue -u "$USER" -h -o '%j')"
while read -r rid; do
  [[ "${rid}" =~ ${PAT} ]] || continue
  name="r4post_${rid}"; name="${name:0:120}"
  if grep -qxF "${name}" <<<"${QUEUED}"; then echo "[queue] ${rid}: controller already queued -- skip"; continue; fi
  if [ -f "${OUTD}/${rid}.status" ] && tail -1 "${OUTD}/${rid}.status" | grep -q "DONE ${rid}"; then echo "[queue] ${rid}: DONE -- skip"; continue; fi
  deps=(); states=()
  for k in 0 1 2; do
    line="$(awk -F'|' -v n="fold${k}_${rid}" '$2==n' <<<"${SACCT}" | sort -t'|' -k1,1n | tail -1)"
    [ -n "${line}" ] || { echo "[queue] ${rid}: no fold${k} job found -- skip"; continue 2; }
    id="${line%%|*}"; st="${line##*|}"; states+=("fold${k}:${st}")
    case "${st}" in PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|SUSPENDED) deps+=("${id}") ;; esac
  done
  dep=""; [ ${#deps[@]} -gt 0 ] && dep="afterany:$(IFS=:; echo "${deps[*]}")"
  echo "[queue] ${rid}: ${states[*]} -> dependency '${dep:-none}'"
  [ "${DRY}" = 1 ] && continue
  RUN_JOB_DEPENDENCY="${dep}" run_job --name "${name}" --gpus 0 --cpus 8 --mem 64G --time 16:00:00 \
      --log "${OUTD}/${rid}.job.log" -- bash scripts/cluster/rung4_lblvor/post_run.sh "${rid}"
  sleep 2
done < "${RUNS}"
