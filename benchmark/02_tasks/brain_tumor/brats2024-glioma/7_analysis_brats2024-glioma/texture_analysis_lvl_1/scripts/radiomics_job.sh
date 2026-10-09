#!/usr/bin/env bash
# Wrapper: per-job PyRadiomics install into $SLURM_TMPDIR (NOT into .venv), then run the given python script.
# usage (via run_job): bash radiomics_job.sh <script.py> [args...]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
cd "$REPO"
if [ -n "${SLURM_TMPDIR:-}" ]; then T="$SLURM_TMPDIR/pyrad"; else T="${SCRATCH:?}/pyrad_tmp"; fi
.venv/bin/python -m pip install -q --no-index --no-deps --target "$T" pyradiomics pykwalify ruamel.yaml docopt python-dateutil 2>&1 | tail -2
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python "$@"
