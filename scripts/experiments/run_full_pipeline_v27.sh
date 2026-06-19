#!/usr/bin/env bash
# Full pipeline for v27a, v27a_bis, synthseg_modeA, synthseg_modeB_em.
# Each phase uses all 4 GPUs in parallel; phases run sequentially (dependency order).
# Logs: /tmp/v27pipeline/<phase>_rN.log
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
CM_ROOT="${REPO_ROOT}/datasets/on-harmony/7_analysis_on-harmony/contrast_manifold"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"

PY="${REPO_ROOT}/.venv/bin/python"
ANALYSIS="${CM_ROOT}/scripts"
DATA_ROOT="${CM_ROOT}/outputs/data"
ORIG_CSV="$DATA_ROOT/original/regional_hist_64/on_harmony_features.csv"
FEAT_CFG="${CM_ROOT}/config/feature_selection.yaml"
SEG_ROOT="${REPO_ROOT}/data/ON-Harmony/derivatives/synthseg_segs"
BIDS_DER="${REPO_ROOT}/data/ON-Harmony/derivatives"

mkdir -p /tmp/v27pipeline
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/v27pipeline/pipeline.log; }

# Expose venv nvidia CUDA libs so TensorFlow (SynthSeg) can find them.
NVIDIA_LIBS=$(find "${REPO_ROOT}/.venv/lib" -mindepth 1 -maxdepth 1 -name "python3.*" \
    -exec find {} -path "*/site-packages/nvidia/*/lib/*.so*" \; 2>/dev/null \
    | xargs -I{} dirname {} | sort -u | tr '\n' ':')

# ── PHASE 1: SynthSeg segs (needed for v27a, v27a_bis, synthseg_modeA) ────────
log "=== PHASE 1: SynthSeg predict (4 GPUs) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "ss_predict_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v27pipeline/ss_predict_r${rank}.log" -- \
        bash -c "export LD_LIBRARY_PATH='${NVIDIA_LIBS}'\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; '${PY}' '${REPO_ROOT}/scripts/run_synthseg_predict.py' --rank ${rank} --world-size 4" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 1 done. Segs: $(find "${SEG_ROOT}" -name '*.nii.gz' 2>/dev/null | wc -l) files"

# ── PHASE 2: v27a generation ─────────────────────────────────────────────────
log "=== PHASE 2: v27a generation (4 GPUs) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "v27a_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v27pipeline/v27a_r${rank}.log" -- \
        "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
        --generator v27a --lhc --n-variants 10 \
        --seg-root "$SEG_ROOT" --rank $rank --world-size 4 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 2 done. v27a: $(find "${BIDS_DER}/synthetic_v27a_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 3: v27a_bis generation ─────────────────────────────────────────────
log "=== PHASE 3: v27a_bis generation (4 GPUs) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "v27a_bis_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v27pipeline/v27a_bis_r${rank}.log" -- \
        "$PY" "${REPO_ROOT}/scripts/generate_synthetic_guidance.py" \
        --generator v27a_bis --lhc --n-variants 10 \
        --seg-root "$SEG_ROOT" --rank $rank --world-size 4 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 3 done. v27a_bis: $(find "${BIDS_DER}/synthetic_v27a_bis_guidance_lhc" -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 4: SynthSeg Mode A generation ──────────────────────────────────────
log "=== PHASE 4: SynthSeg Mode A generation (4 GPUs) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "ss_modeA_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v27pipeline/ss_modeA_r${rank}.log" -- \
        bash -c "export LD_LIBRARY_PATH='${NVIDIA_LIBS}'\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; '${PY}' '${REPO_ROOT}/scripts/generate_synthseg_comparison.py' --mode modeA --n-variants 10 --rank ${rank} --world-size 4" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 4 done. synthseg_modeA: $(find "${BIDS_DER}/synthseg_modeA" -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 5: SynthSeg Mode B (EM) generation ─────────────────────────────────
log "=== PHASE 5: SynthSeg Mode B (EM) generation (4 GPUs) ==="
PIDS=()
for rank in 0 1 2 3; do
    run_job --name "ss_modeB_r${rank}" --gpus 1 --slot $rank --wait \
        --log "/tmp/v27pipeline/ss_modeB_r${rank}.log" -- \
        bash -c "export LD_LIBRARY_PATH='${NVIDIA_LIBS}'\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; '${PY}' '${REPO_ROOT}/scripts/generate_synthseg_comparison.py' --mode modeB --n-variants 10 --rank ${rank} --world-size 4" &
    PIDS+=($!)
done
wait "${PIDS[@]}"
log "Phase 5 done. synthseg_modeB_em: $(find "${BIDS_DER}/synthseg_modeB_em" -name '*.nii.gz' 2>/dev/null | wc -l)/1650 files"

# ── PHASE 6: Feature extraction (4 versions in parallel) ──────────────────────
log "=== PHASE 6: Feature extraction (4 parallel slots) ==="
mkdir -p \
  "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64" \
  "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64" \
  "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64"

PIDS=()
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/extract_v27a.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" --mode synthetic \
    --synth-root "$BIDS_DER/synthetic_v27a_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features.csv" \
    --n-workers 56 &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/extract_v27a_bis.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" --mode synthetic \
    --synth-root "$BIDS_DER/synthetic_v27a_bis_guidance_lhc" \
    --output-csv "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features.csv" \
    --n-workers 56 &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/extract_ss_modeA.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" --mode synthetic \
    --synth-root "$BIDS_DER/synthseg_modeA" \
    --output-csv "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features.csv" \
    --n-workers 56 &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/extract_ss_modeB.log" -- \
    "$PY" "$ANALYSIS/extract_features_regional_hist.py" --mode synthetic \
    --synth-root "$BIDS_DER/synthseg_modeB_em" \
    --output-csv "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features.csv" \
    --n-workers 56 &
PIDS+=($!)

wait "${PIDS[@]}"
log "Phase 6 done."

# ── PHASE 7: Normalize + feature selection (4 parallel, CPU-only) ──────────────
log "=== PHASE 7: Normalize ==="
PIDS=()
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/normalize_v27a.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_v27a_guidance_lhc/regional_hist_64/synthetic_v27a_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/normalize_v27a_bis.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_v27a_bis_guidance_lhc/regional_hist_64/synthetic_v27a_bis_guidance_lhc_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/normalize_ss_modeA.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_synthseg_modeA/regional_hist_64/synthetic_synthseg_modeA_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" &
PIDS+=($!)

run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/normalize_ss_modeB.log" -- \
    "$PY" "$ANALYSIS/normalize_combined.py" \
    --original_csv  "$ORIG_CSV" \
    --synthetic_csv "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features.csv" \
    --output_original  "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv" \
    --output_synthetic "$DATA_ROOT/synthetic_synthseg_modeB_em/regional_hist_64/synthetic_synthseg_modeB_em_features_normalized_combined.csv" \
    --feature_config "$FEAT_CFG" &
PIDS+=($!)

wait "${PIDS[@]}"
log "Phase 7 done."

# ── PHASE 8: Full manifold analysis (UMAP, PRDC, Vendi) ──────────────────────
log "=== PHASE 8: Analysis ==="
PIDS=()
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/analysis_v27a.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" --mask-type regional_hist_64 --only v27a_guidance_lhc_r1 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/analysis_v27a_bis.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" --mask-type regional_hist_64 --only v27a_bis_guidance_lhc_r1 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/analysis_ss_modeA.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" --mask-type regional_hist_64 --only synthseg_modeA_r1 &
PIDS+=($!)
run_job --gpus 0 --slot 0 --wait --log "/tmp/v27pipeline/analysis_ss_modeB.log" -- \
    "$PY" "$ANALYSIS/run_all_analysis.py" --mask-type regional_hist_64 --only synthseg_modeB_em_r1 &
PIDS+=($!)
wait "${PIDS[@]}"
log "Phase 8 done."

log "=== ALL PHASES COMPLETE ==="
