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

Formal mapping used by the v15 operator:

Given a voxel intensity x_i in [0,1], define spatially varying quantile edges
{q_k(i)}_{k=0..K} from interpolated grid-local quantiles, with q_0(i)=0 and q_K(i)=1.
The chunk assignment is

$$
c_i = \sum_{k=1}^{K} k \cdot \mathbf{1}\big[q_{k-1}(i) < x_i \le q_k(i)\big].
$$

For each chunk k, draw an independent random target

$$
\mu_k \sim \mathcal{U}(0,1), \quad k=1,\dots,K,
$$

and synthesize

$$
x_i^{\text{synth}} = \mu_{c_i}.
$$

Background is preserved by a tissue mask m_i = 1[x_i > \tau] (with \tau about 0.01):

$$
\widetilde{x}_i^{\text{synth}} = m_i \cdot x_i^{\text{synth}}.
$$

Because {\mu_k} is unsorted, the mapping is explicitly non-monotonic and permits contrast inversion.

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

Formal consistency objective (v13 inherited by v15):

Let p_i = sigma(z_i^{raw}) and q_i = sigma(z_i^{synth}) be Bernoulli probabilities from raw and synthetic logits at voxel i.
The memory-lean Bernoulli KL term is

$$
\mathrm{KL}(p_i\|q_i) = p_i\log\frac{p_i}{q_i} + (1-p_i)\log\frac{1-p_i}{1-q_i}.
$$

Aggregated consistency loss:

$$
\mathcal{L}_{cons} = \frac{1}{N}\sum_{i=1}^{N} \mathrm{KL}(\mathrm{stopgrad}(p_i)\|q_i).
$$

Optional Jensen-Shannon form (symmetric variant):

$$
\mathrm{JS}(p_i,q_i) = \frac{1}{2}\mathrm{KL}(p_i\|m_i) + \frac{1}{2}\mathrm{KL}(q_i\|m_i),
\quad m_i = \frac{1}{2}(p_i+q_i).
$$

Total segmenter objective in consistency mode:

$$
\mathcal{L}_{total} = \mathcal{L}_{sup}(q, y) + \lambda_{cons}\,\mathcal{L}_{cons}.
$$

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
