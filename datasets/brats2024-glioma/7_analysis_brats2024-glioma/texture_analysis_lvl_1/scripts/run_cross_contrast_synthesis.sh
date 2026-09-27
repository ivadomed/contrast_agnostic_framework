#!/usr/bin/env bash
# Feature-level (learned cross-contrast synthesis) containment pilot driver, BraTS2024-glioma.
# Runs entirely via Slurm on Vulcan (never the login node). Usage:
#   bash run_cross_contrast_synthesis.sh smoke   # cache 4/2/2 patients, 50-iter train, 2-patient infer
#   bash run_cross_contrast_synthesis.sh train    # submit the 4 full per-source training jobs (parallel)
#   bash run_cross_contrast_synthesis.sh infer    # inference on all 70 eval patients (run after all 4 finish)
#   bash run_cross_contrast_synthesis.sh summary  # pooled table/plot
set -euo pipefail

LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
S="${LVL1}/scripts"
LOGDIR="${LVL1}/outputs/logs"
mkdir -p "${LOGDIR}"

source "${REPO}/datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"

cmd="${1:-}"

case "${cmd}" in
  smoke)
    echo "== smoke: build tiny cache =="
    run_job --name synth_smoke_cache --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --wait \
        --log "${LOGDIR}/smoke_cache.log" -- "${PY}" "${S}/build_cache.py" --smoke
    cat "${LOGDIR}/smoke_cache.log"

    echo "== smoke: 50-iter train (t1n) =="
    run_job --name synth_smoke_train --gpus 1 --cpus 8 --mem 32G --time 00:30:00 --wait \
        --log "${LOGDIR}/smoke_train.log" -- "${PY}" "${S}/train_synth.py" --source t1n --iters 50 --smoke
    cat "${LOGDIR}/smoke_train.log"

    echo "== smoke: infer on 2 patients (needs all 4 sources trained -- run smoke train for all 4 first if not already) =="
    for src in t1c t2w t2f; do
      run_job --name "synth_smoke_train_${src}" --gpus 1 --cpus 8 --mem 32G --time 00:30:00 --wait \
          --log "${LOGDIR}/smoke_train_${src}.log" -- "${PY}" "${S}/train_synth.py" --source "${src}" --iters 50 --smoke
    done
    run_job --name synth_smoke_infer --gpus 1 --cpus 8 --mem 32G --time 00:30:00 --wait \
        --log "${LOGDIR}/smoke_infer.log" -- "${PY}" "${S}/infer_synth.py" --smoke --n-qc-patients 2
    cat "${LOGDIR}/smoke_infer.log"
    ;;

  cache)
    echo "== build full cache (100 train + 20 val + 70 eval) =="
    run_job --name synth_cache --gpus 0 --cpus 8 --mem 64G --time 02:00:00 --wait \
        --log "${LOGDIR}/cache.log" -- "${PY}" "${S}/build_cache.py"
    cat "${LOGDIR}/cache.log"
    ;;

  train)
    echo "== submitting 4 full training jobs (parallel, ~4000 iters each) =="
    for src in t1n t1c t2w t2f; do
      run_job --name "synth_train_${src}" --gpus 1 --cpus 8 --mem 64G --time 08:00:00 \
          --log "${LOGDIR}/train_${src}.log" -- "${PY}" "${S}/train_synth.py" --source "${src}" --iters 4000
      sleep 5
    done
    echo "Submitted. Check squeue -u \$USER (every few minutes, not in a tight loop)."
    ;;

  infer)
    echo "== inference on all 70 eval patients (all 4 models) =="
    run_job --name synth_infer --gpus 1 --cpus 8 --mem 64G --time 06:00:00 --wait \
        --log "${LOGDIR}/infer.log" -- "${PY}" "${S}/infer_synth.py" --n-qc-patients 2
    cat "${LOGDIR}/infer.log"
    ;;

  summary)
    echo "== pooled summary (CPU) =="
    run_job --name synth_summary --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
        --log "${LOGDIR}/summary.log" -- "${PY}" "${S}/synthesis_vs_fill_swap_summary.py"
    cat "${LOGDIR}/summary.log"
    ;;

  *)
    echo "usage: bash run_cross_contrast_synthesis.sh {smoke|cache|train|infer|summary}" >&2
    exit 1
    ;;
esac
