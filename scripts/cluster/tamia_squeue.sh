#!/usr/bin/env bash
# Read-only job-queue check on TamIA, relayed through Vulcan.
# Wrapping this in a fixed script (rather than a raw `ssh` call) lets it be
# safely allowlisted: the remote command is hardcoded here, not passed
# through as free-form shell text, so there's no way to smuggle a mutating
# command past the permission check the way a `Bash(ssh host "squeue*)`
# prefix-allow rule could be (anything appended after "squeue" inside the
# quotes would still match that prefix).
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=15 tamia.alliancecan.ca \
  "squeue -u paulh -o '%.10i %.9P %.30j %.8T %.10M %.6D %R'"
