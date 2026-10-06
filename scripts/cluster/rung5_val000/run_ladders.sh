#!/bin/bash
# CPU job body: regenerate the causal-ablation ladders whose rung 5 now points at the val000 retrain, through each
# dataset's OWN ladder script (no logic here). Submit through run_job, e.g.
#   run_job --name r5_ladders --gpus 0 --cpus 4 --mem 32G --time 01:30:00 --log $SCRATCH/rung5_val000/ladders.log --wait -- \
#       bash scripts/cluster/rung5_val000/run_ladders.sh abdomen_t2spir mandible pelvis_ct pelvis_mri breast_t2w
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export PROJECT_ROOT="$PWD" RUN_JOB_INLINE=1
T=benchmark/02_tasks
rc=0
for k in "$@"; do
  echo "=== ${k} $(date +%T)"
  case "$k" in
    abdomen_t2spir) ( source $T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env_t2spir.sh
                      .venv/bin/python $T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_35_ladder_summary_t2spir.py ) ;;
    mandible)       ( source $T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/00_utils/env.sh
                      .venv/bin/python $T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_05_ladder_summary.py ) ;;
    pelvis_ct)      .venv/bin/python $T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_10_ladder_summary_ct.py ;;
    pelvis_mri)     .venv/bin/python $T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_11_ladder_summary_mri.py ;;
    breast_t1wce)   .venv/bin/python $T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_10_ladder_summary_t1wce.py ;;
    breast_t2w)     .venv/bin/python $T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_11_ladder_summary_t2w.py ;;
    glioma_t1n)     .venv/bin/python $T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py ;;
    glioma_t2w)     .venv/bin/python $T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py ;;
    glioma_t2f)     .venv/bin/python $T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_16_ladder_summary_t2f.py ;;
    glioma_t1c)     .venv/bin/python $T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_33_ladder_summary_t1c.py ;;
    ms_t1w)         .venv/bin/python $T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_18_ladder_summary_t1w.py ;;
    *) echo "unknown ladder key $k"; false ;;
  esac || { echo "!!! ${k} FAILED"; rc=1; }
done
echo "run_ladders rc=${rc}"; exit ${rc}
