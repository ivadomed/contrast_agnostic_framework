#!/usr/bin/env bash
# Evaluate ALL chaos models on the chimeras (both contrasts, all methods, kidney on the
# US region). Methods×folds run in PARALLEL (CPU jobs; each fans 4 folds, all methods
# backgrounded). See 06_06_evaluate_chimera_run.sh.
#   bash 06_07_evaluate_chimera_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "=== TRUSTED CHIMERA eval | t1in + t2spir | kidney on US region ==="

# t1in
for r in chaos_t1in_baseline_20260614_153230 \
         chaos_t1in_v26_6_2_train050_val100_20260615_213615 \
         chaos_t1in_synthseg_EM_train100_val000_20260611_120000 \
         chaos_t1in_synthseg_noEM_train100_val000_20260611_120000 \
         chaos_t1in_auglab_default_20260611_120000 \
         chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514; do
  bash "${HERE}/06_06_evaluate_chimera_run.sh" "$r" all &
done

# t2spir (contrast vars pre-exported in a subshell so env.sh routes to chimera/t2spir)
for r in chaos_t2spir_baseline_20260620_111146 \
         chaos_t2spir_v26_6_2_train050_val100_20260620_112122 \
         chaos_t2spir_synthseg_EM_20260620_112357 \
         chaos_t2spir_synthseg_noEM_20260620_112515 \
         chaos_t2spir_auglab_default_20260620_112240 \
         chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633; do
  ( export CHAOS_TRAINING_CONTRAST=t2spir CHAOS_DATASET_ID=61 CHAOS_DS_NAME=Dataset061_CHAOS_MR_T2spir
    bash "${HERE}/06_06_evaluate_chimera_run.sh" "$r" all ) &
done
wait
echo "[$(date '+%H:%M:%S')] all chimera evaluations done. Aggregate: bash 06_08_aggregate_chimera.sh"
