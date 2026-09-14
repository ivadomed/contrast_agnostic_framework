#!/usr/bin/env bash
# Read-only Slurm accounting check on TamIA, relayed through Vulcan.
# Usage: tamia_sacct.sh <jobid[,jobid...]>
# Job-id argument is validated (digits/commas only) before being passed to
# ssh, so this stays a pure read-only lookup regardless of the argument —
# see tamia_squeue.sh for why this is a script and not an allowlisted raw
# `ssh` prefix pattern.
set -euo pipefail
JOBIDS="${1:?usage: tamia_sacct.sh <jobid[,jobid...]>}"
if [[ ! "$JOBIDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "error: job id argument must be digits/commas only, got: $JOBIDS" >&2
  exit 1
fi
ssh -o BatchMode=yes -o ConnectTimeout=15 tamia.alliancecan.ca \
  "sacct -j $JOBIDS --format=JobID,JobName%40,State,Elapsed,ExitCode --noheader"
