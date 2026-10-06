#!/usr/bin/env bash
# Reproduce everything: submit one GPU job per dataset/contrast (fire-and-forget), then run
# `bash scripts/run_aggregate.sh` once they are all finished (outputs: data/ngf_*.csv).
# usage: bash scripts/run_all.sh [N_SCANS=20] [N_DRAWS=5] [VARIANTS=noblur,asblur]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-20}"; K="${2:-5}"; V="${3:-noblur,asblur}"
# key : wall-time (generous; measured in the smoke runs, see ngf_summary.md)
declare -A T=(
 [brats2024-glioma/t1n]=01:30:00 [brats2024-glioma/t2w]=01:30:00 [brats2024-glioma/t2f]=01:30:00 [brats2024-glioma/t1c]=01:30:00
 [open-ms/flair]=01:30:00 [open-ms/t1w]=01:30:00 [ispy2/t1wce]=01:30:00 [ispy2/t2w]=01:30:00
 [chaos/t1in]=01:30:00 [chaos/t2spir]=01:30:00 [on-harmony/t1w]=01:30:00 [on-harmony/t2w]=01:30:00 [on-harmony/dwi]=01:30:00
 [toothfairy2/cbct]=02:00:00 [totalseg-pelvic/ct]=02:00:00 [totalseg-pelvic/mri]=01:30:00 )
for k in "${!T[@]}"; do
  MAXV=12000000; [ "$k" = "totalseg-pelvic/ct" ] && MAXV=8000000   # SynthSeg/PALETTE GPU memory: CT volumes are cropped to <=8M voxels, all others <=12M
  bash "${HERE}/run_key.sh" "$k" "${T[$k]}" --n-scans "$N" --n-draws "$K" --variants "$V" --max-vox "$MAXV"
  sleep 3     # space out sbatch calls
done
