#!/usr/bin/env bash
# Run from the login node (has internet access) after Slurm jobs finish, to
# push WandB runs that were logged offline (run_job_slurm.sh sets
# WANDB_MODE=offline, since compute nodes have no internet) up to the cloud.
#
# NOTE: `wandb sync --sync-all` is the standard flag for "sync every
# not-yet-synced run under this wandb/ dir" across recent wandb releases
# (installed here: wandb==0.25.1) — not re-verified live via `wandb --help`
# in this pass because the project .venv's python interpreter is currently
# broken (dangling symlink from the machine this repo was copied from; see
# CLAUDE.md). Re-check once that's fixed.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -d wandb ]; then
    echo "No wandb/ directory at ${PROJECT_ROOT} — nothing to sync."
    exit 0
fi

.venv/bin/wandb sync --sync-all wandb
