# Architecture and Rationale

## 1. Objective
Build a contrast-agnostic 3D brain tumor segmenter that can generalize across MRI contrasts without requiring paired target-contrast supervision.

The current system-level strategy is:
- Use a guidance generator to create synthetic contrast perturbations that preserve anatomy while diversifying intensity semantics.
- Train the segmenter to remain anatomically consistent across raw and synthesized inputs.

## 2. Current SOTA Candidate: v15
v15 combines:
- v8-style spatially localized grid chunking (preserves macro-structural sharpness),
- v13-style non-monotonic intensity remapping (allows contrast inversion),
- explicit background masking (prevents gray background hallucination),
- inherited anisotropic degradation and consistency regularization in segmenter training.

## 3. End-to-End Training Pipeline

### 3.1 Input and normalization
Input volumes are normalized to x in [0,1] with shape (B, C, D, H, W).

### 3.2 Guidance target synthesis (v15)
Core operator: non-monotonic grid chunking.

For each batch:
1. Partition each volume into a coarse grid (default 4x4x4).
2. Compute local quantile edges per grid block.
3. Trilinearly upsample local thresholds to dense thresholds over full resolution.
4. Assign each voxel to a chunk index from these spatially varying thresholds.
5. Sample unsorted random chunk targets u_k ~ U(0,1).
6. Map each chunk index directly to its sampled target.
7. Apply strict background preservation using a tissue mask threshold (~0.01).

This allows non-monotonic mappings where intensity order can invert, while still respecting local spatial anatomy.

### 3.3 Generator path
The generator is a 3D synthesis network optimized with a composite objective:
- histogram/distribution alignment,
- edge-aware structure retention,
- smoothness regularization,
- range constraints,
- guidance consistency terms.

Version-gated augmentations and target generation are integrated through compiled wrappers to keep execution deterministic and fast enough for iterative research.

### 3.4 Segmenter path
The segmenter is a 3D U-Net style model trained on synthesized inputs (and baseline modes where applicable), with Dice+CE supervision and version-gated robustness terms.

v15 segmenter inherits:
- anisotropic degradation branch (clinical thick-slice simulation lineage from v11+),
- consistency regularization lineage from v13.

## 4. Unsupervised Consistency Regularization (Dual-Pass via Batch Concatenation)

### 4.1 Motivation
The consistency objective enforces prediction invariance to intensity-domain perturbations produced by the generator. This prevents the segmenter from overfitting to a narrow synthetic style.

### 4.2 Implementation pattern
Instead of two independent forwards:
- model(x_raw)
- model(x_synth)

the system uses one concatenated forward:
1. x_combined = cat([x_raw, x_synth], dim=0)
2. logits_combined = segmenter(x_combined)
3. split logits into raw and synth halves
4. supervised segmentation loss on synthetic branch
5. consistency KL-style penalty between detached raw predictions and synthetic predictions
6. total_loss = supervised + lambda_consistency * consistency

### 4.3 Why this matters
- Preserves mathematical intent (invariance to contrast perturbation).
- Reduces launch overhead versus sequential dual forwards.
- Improves throughput stability in compiled 3D training.

## 5. Why v15 is the current architecture direction
v15 is the first design that jointly addresses the two dominant prior failure classes:
- Over-monotonic mappings that cannot invert contrast semantics.
- Spatially over-smoothed mappings that wash out macro-structural contrast.

By combining local spatial chunking with non-monotonic targets and strict background control, v15 preserves structure while increasing contrast diversity in anatomically plausible regions.

## 6. Practical Design Constraints
The architecture is intentionally shaped by known physical and computational limits:
- Physics asymmetry: T2w -> T1w boundary synthesis remains fundamentally harder than T1w -> other contrasts.
- 3D compute budget: generator + histogram + multi-loss training is inherently expensive; compilation and vectorization are mandatory, not optional.
- Reproducibility: Hydra + Lightning + version-gated operators are required to keep experimental claims auditable across versions.

## 7. Deployment-facing interpretation
For downstream users, the practical contract is:
- Use version-matched generator and segmenter settings.
- Preserve background masking and mask-safe augmentations.
- Treat v15 as the current best candidate for robust cross-contrast transfer, pending further ensemble and external-cohort validation.
