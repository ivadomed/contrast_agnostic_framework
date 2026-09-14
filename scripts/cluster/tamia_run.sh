#!/usr/bin/env bash
# General-purpose runner for TamIA, relayed through Vulcan, always cd'd into the repo
# (see CLAUDE.md: "Remote commands over the TamIA relay MUST carry an explicit cd").
#
# This exists so a TamIA call is always the same fixed local command
# (`bash scripts/cluster/tamia_run.sh ...`), regardless of how the remote command
# is built. Constructing the ssh invocation inline (REPO=... ; CMD=... ; ssh ...)
# makes the Bash-tool command start with the variable assignment, not "ssh" -- which
# defeats prefix-based permission allowlisting no matter what ssh flags are used.
# Routing everything through this one entrypoint keeps a single allowlist rule
# working regardless of the remote command's own complexity/quoting.
#
# Usage:
#   bash scripts/cluster/tamia_run.sh "squeue -u \$USER"
#   bash scripts/cluster/tamia_run.sh <<'EOF'
#   multi
#   line
#   remote command
#   EOF
#
# Overall wall-clock budget (connection + remote execution) defaults to 120s;
# override with TAMIA_RUN_TIMEOUT=<seconds>. This is internal so callers never
# need to prefix `timeout N` themselves -- that prefix breaks the fixed-prefix
# permission match this script exists to provide (see above).
set -euo pipefail
REPO='/project/aip-jcohen/paulh/mri_synthesis_project'
RUN_TIMEOUT="${TAMIA_RUN_TIMEOUT:-120}"

if [ "$#" -gt 0 ]; then
  REMOTE_CMD="$*"
else
  REMOTE_CMD=$(cat)
fi

FULL_CMD="cd $REPO && { $REMOTE_CMD
}"

timeout "$RUN_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=20 tamia.alliancecan.ca \
  "bash -lc $(printf '%q' "$FULL_CMD")"
