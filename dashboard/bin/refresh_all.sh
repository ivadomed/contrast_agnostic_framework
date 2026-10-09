#!/usr/bin/env bash
# Orchestrator run on vulcan (by cron or manually) to refresh the dashboard's
# view of both clusters:
#   1. Scans vulcan itself (local filesystem + local squeue/diskusage/sshare).
#   2. Pulls killarney's own status_killarney.json (produced by killarney's
#      OWN cron running the same scan_cluster.py locally there) over the
#      persistent SSH control socket. See sync_remote_status.sh for details
#      and the one-time interactive setup it requires.
#   3. Writes combined dashboard/status_combined.json for convenience (the
#      Streamlit app can also just read the two per-cluster files directly).
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/aip-jcohen/paulh/mri_synthesis_project}"
BIN_DIR="$PROJECT_ROOT/dashboard/bin"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"

echo "[refresh_all] $(date -u +%FT%TZ) scanning vulcan locally..."
python3 "$BIN_DIR/scan_cluster.py" --project-root "$PROJECT_ROOT" --cluster vulcan \
    --out "$DASHBOARD_DIR/status_vulcan.json"

echo "[refresh_all] $(date -u +%FT%TZ) syncing killarney status..."
bash "$BIN_DIR/sync_remote_status.sh" "$PROJECT_ROOT"

echo "[refresh_all] $(date -u +%FT%TZ) combining..."
python3 "$BIN_DIR/combine_status.py" --project-root "$PROJECT_ROOT"

echo "[refresh_all] $(date -u +%FT%TZ) done."
