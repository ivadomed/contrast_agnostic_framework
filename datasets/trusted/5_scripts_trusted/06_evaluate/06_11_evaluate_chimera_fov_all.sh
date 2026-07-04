#!/usr/bin/env bash
# Evaluate ALL chaos models on the chimeras with the FOV-restricted, side-aware kidney
# metric (both contrasts, all methods). See 06_10_evaluate_chimera_fov_run.sh.
# Methods run in parallel; each fans 4 folds. Results land in the chimera_fov/ namespace.
#   bash 06_11_evaluate_chimera_fov_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "=== TRUSTED CHIMERA FOV eval | t1in + t2spir | side-aware kidney over CHAOS FOV ==="

# t1in
for r in chaos_t1in_baseline_20260614_153230 \
         chaos_t1in_v26_6_2_train050_val100_20260615_213615 \
         chaos_t1in_synthseg_EM_train100_val000_20260611_120000 \
         chaos_t1in_synthseg_noEM_train100_val000_20260611_120000 \
         chaos_t1in_auglab_default_20260611_120000 \
         chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514; do
  bash "${HERE}/06_10_evaluate_chimera_fov_run.sh" "$r" all &
done

# t2spir (contrast vars pre-exported so env.sh routes to chimera/t2spir)
for r in chaos_t2spir_baseline_20260620_111146 \
         chaos_t2spir_v26_6_2_train050_val100_20260620_112122 \
         chaos_t2spir_synthseg_EM_20260620_112357 \
         chaos_t2spir_synthseg_noEM_20260620_112515 \
         chaos_t2spir_auglab_default_20260620_112240 \
         chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633; do
  ( export CHAOS_TRAINING_CONTRAST=t2spir CHAOS_DATASET_ID=61 CHAOS_DS_NAME=Dataset061_CHAOS_MR_T2spir
    bash "${HERE}/06_10_evaluate_chimera_fov_run.sh" "$r" all ) &
done
wait
echo "[$(date '+%H:%M:%S')] all chimera-FOV evaluations done → chimera_fov/ namespace."
