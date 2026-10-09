#!/bin/bash
# Persistent loop that re-runs scan_cluster.py on killarney every 15 minutes.
# killarney's login node crontab is disabled for regular users
# (/etc/cron.allow denies `paulh`), so periodic refresh here uses a
# self-looping detached process instead of cron. Launch with:
#   setsid nohup dashboard/bin/killarney_scan_loop.sh >> dashboard/logs/scan_loop.log 2>&1 < /dev/null &
#   disown
# Check it's alive with: pgrep -af killarney_scan_loop
# Stop it with:          pkill -f killarney_scan_loop.sh
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/project/aip-jcohen/paulh/mri_synthesis_project}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"   # 15 minutes
PY="${PY:-python3}"

cd "$PROJECT_ROOT" || exit 1
mkdir -p dashboard/logs

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$ts] running scan_cluster.py ..."
  "$PY" dashboard/bin/scan_cluster.py \
    --project-root "$PROJECT_ROOT" \
    --cluster killarney \
    --out dashboard/status_killarney.json \
    >> dashboard/logs/scan_loop.log 2>&1
  rc=$?
  ts2="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$ts2] scan_cluster.py exit=$rc; sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done
