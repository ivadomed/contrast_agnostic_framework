#!/usr/bin/env bash
# Runs on vulcan. Pulls killarney's freshly-scanned status_killarney.json into
# vulcan's local dashboard/ directory so the Streamlit app (which only ever
# reads local files) can see both clusters.
#
# Alliance Canada enforces interactive MFA on every fresh SSH login, so this
# script CANNOT authenticate on its own. Instead it relies on an existing
# multiplexed SSH ControlMaster socket (see ~/.ssh/config, already configured
# with `ControlMaster auto` + `ControlPersist 30d` for killarney) that the
# user opens once, interactively, by running:
#
#     ssh killarney.alliancecan.ca true
#
# and completing the MFA prompt. Every call to this script for the next 30
# days of inactivity reuses that socket silently. If the socket has expired
# or was never opened, this script fails soft: it leaves the last-known copy
# of status_killarney.json in place and stamps a *_sync_failed marker so the
# dashboard can surface a staleness warning instead of crashing.
set -uo pipefail

PROJECT_ROOT="${1:-/project/aip-jcohen/paulh/mri_synthesis_project}"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
REMOTE_HOST="killarney.alliancecan.ca"
REMOTE_PATH="$PROJECT_ROOT/dashboard/status_killarney.json"
LOCAL_PATH="$DASHBOARD_DIR/status_killarney.json"
SYNC_META="$DASHBOARD_DIR/sync_meta.json"

mkdir -p "$DASHBOARD_DIR"

# Quick liveness check on the control socket (2s timeout, no fallback prompt).
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" true 2>/tmp/sync_ssh_err_$$; then
    if scp -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH.tmp" 2>/tmp/sync_scp_err_$$; then
        mv "$LOCAL_PATH.tmp" "$LOCAL_PATH"
        python3 -c "
import json, datetime
json.dump({'last_sync_ok': True, 'last_sync_at': datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open('$SYNC_META', 'w'))
"
        echo "[sync] pulled fresh status_killarney.json from $REMOTE_HOST"
        rm -f "/tmp/sync_ssh_err_$$" "/tmp/sync_scp_err_$$"
        exit 0
    fi
fi

# Soft-fail path: SSH control socket not available (needs one interactive
# MFA login) or scp failed. Keep the stale local copy and record the failure
# so the dashboard can show "killarney data last refreshed N ago".
python3 -c "
import json, datetime
json.dump({'last_sync_ok': False, 'last_sync_attempt_at': datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open('$SYNC_META', 'w'))
"
echo "[sync] WARNING: could not reach $REMOTE_HOST (no live SSH control socket)."
echo "[sync] Run 'ssh $REMOTE_HOST true' interactively once to open a 30-day-persistent connection, then retry."
echo "[sync] Dashboard will show the last-known killarney status with a staleness badge."
rm -f "/tmp/sync_ssh_err_$$" "/tmp/sync_scp_err_$$"
exit 0
