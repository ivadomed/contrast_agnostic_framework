#!/usr/bin/env bash
# General-purpose runner for Killarney (directly reachable, no relay needed unlike
# TamIA), always cd'd into the repo (see CLAUDE.md: remote commands should carry
# an explicit cd rather than relying on the login shell's default directory).
#
# Mirrors scripts/cluster/tamia_run.sh: gives every Killarney call the same fixed
# local command (`bash scripts/cluster/killarney_run.sh ...`) regardless of how
# the remote command is built, so a single allowlist rule covers it instead of
# every possible local prefix (env vars, `timeout N`, etc.) that would otherwise
# defeat prefix-based permission matching.
#
# Usage:
#   bash scripts/cluster/killarney_run.sh "squeue -u \$USER"
#   bash scripts/cluster/killarney_run.sh <<'EOF'
#   multi
#   line
#   remote command
#   EOF
#
# Overall wall-clock budget (connection + remote execution) defaults to 120s;
# override with KILLARNEY_RUN_TIMEOUT=<seconds>.
set -euo pipefail
REPO='/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project'
RUN_TIMEOUT="${KILLARNEY_RUN_TIMEOUT:-120}"

if [ "$#" -gt 0 ]; then
  REMOTE_CMD="$*"
else
  REMOTE_CMD=$(cat)
fi

FULL_CMD="cd $REPO && { $REMOTE_CMD
}"

timeout "$RUN_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=20 killarney.alliancecan.ca \
  "bash -lc $(printf '%q' "$FULL_CMD")"
