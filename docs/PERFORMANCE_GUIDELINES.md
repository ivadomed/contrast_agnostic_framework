# Performance Guidelines

This document defines strict engineering rules for maintaining training throughput and correctness in the 3D MRI pipeline.

## 1. Performance SLOs
- Generator-only target: about 1.5 minutes per epoch (reference: v13 benchmark ~67/67 in 0:01:36).
- Segmenter with consistency target: steady-state about 15 seconds per epoch for 33 batches.
- Segmenter baseline target: single-digit seconds per epoch where hardware permits.

If a change violates SLOs, treat it as a regression until profiler evidence proves otherwise.

## 2. Non-Negotiable Rules

### Priority Rule 0: Allocate generator compute to T1w-source runs
- Treat T1w-source generator training as the default compute path.
- Do not spend routine generator slots on T2w-source training unless explicitly approved for a narrowly scoped experiment.

Rationale: this maximizes return on limited slots and aligns with the established T1w-first strategy.

### Rule 1: Keep histogram and guidance paths vectorized
- No Python loops over batch or chunk dimensions in hotspot operators.
- Use batched tensor operations (searchsorted, quantiles, indexing, interpolation).

Rationale: host loops and scalar extraction stall GPU dispatch and destroy throughput.

### Rule 2: Use separable 1D convolutions for 3D Gaussian blur
- Replace dense 3D kernels with depth-height-width 1D separable passes where equivalent.

Rationale: reduces effective complexity from O(N^3) style kernel growth to O(3N) style passes.

Hard requirement:
- Any new 3D blur implementation must preserve the separable O(3N) formulation unless profiler evidence justifies an exception.

### Rule 3: Avoid forced memory-format thrashing
- Do not add gratuitous clone() or contiguous() in critical paths.
- Preserve channels_last_3d-compatible flow unless a proven kernel requires conversion.

Rationale: aten::copy_ dominated previous traces; memory churn can consume 30% of runtime.

### Rule 4: Compile wrappers around stable graph boundaries
- Keep compiled wrappers stateless and narrow.
- Prefer compiling forward+loss units with stable signatures.
- Avoid wrapping heavy mutable module state that causes graph churn.

Rationale: stable compile boundaries reduce Dynamo/Inductor recompilation overhead.

### Rule 5: Keep augmentations on GPU in training hot paths
- High-frequency augmentation primitives must run in batched GPU form.
- Avoid CPU dataloader-side stochastic transforms for expensive operations.

Rationale: prevents CPU starvation and host-device synchronization bottlenecks.

### Rule 6: Use batch concatenation for consistency regularization
- Compute raw and synthetic predictions in one concatenated forward pass.
- Do not reintroduce sequential dual forward passes unless justified by profiler data.

Rationale: preserves invariance objective while reducing launch overhead.

### Rule 7: Use memory-lean Bernoulli KL for consistency
- Keep consistency loss in direct Bernoulli form.
- Do not reconstruct temporary two-channel [p, 1-p] distributions at full 3D resolution.

Rationale: avoids avoidable OOM risk and memory pressure.

### Rule 8: Preserve mask-safe interpolation semantics
- Labels/masks must always use nearest-neighbor resampling for geometric transforms.
- Images may use smooth interpolation, but masks cannot.

Rationale: incorrect mask interpolation can collapse Dice even when CE appears to improve.

### Rule 9: Maintain deterministic execution contracts
- Eliminate nondeterministic Python control branching in compiled/distributed-critical paths.
- Use tensor-native RNG and consistent augmentation return signatures.

Rationale: avoids silent divergence and runtime unpack failures.

### Rule 10: Version-gate behavior explicitly
- New generation logic must be behind explicit version gates.
- Do not alter behavior of prior versions by side effect.

Rationale: protects comparability and reproducibility across historical runs.

## 3. Profiling-Driven Development Protocol
Any performance claim must include:
1. Profiler context (tool, active batches, hardware).
2. Before/after top operators with percent-of-total time.
3. Copy/clone event trend.
4. End-to-end epoch timing, not only micro-benchmark kernels.

Do not merge speed changes based only on intuition.

## 4. Correctness Before Speed Checklist
Run this checklist before accepting an optimization:
1. Metrics sanity: no unexplained collapse in in-domain, OOD mean, or OOD worst.
2. Label integrity: masks visually and numerically preserved after transforms.
3. Data contract integrity: augmentation outputs keep expected tuple/tensor schema.
4. Version isolation: old versions reproduce previous behavior.

## 5. Known High-Cost Components
Expect substantial cost from:
- 3D generator forward/backward.
- DifferentiableHistogram3D passes.
- Multi-term loss stacks.

Interpretation:
- Some overhead is architectural and cannot be removed without algorithmic simplification.

## 6. Approved Optimization Directions
Prioritized future work:
1. Histogram dimensionality reduction with calibration checks.
2. Loss-kernel fusion to reduce sequential launch overhead.
3. Generator capacity trims with quality guardrails.
4. Compile-mode tuning with teardown-safe settings.

## 7. Anti-Patterns to Reject in Code Review
Reject changes that:
- Reintroduce per-sample Python loops in 3D hotspot kernels.
- Add clone()/contiguous() without benchmark evidence.
- Mix mask and image interpolation modes in one unsafe path.
- Expand logging/media output in inner loops without gating.
- Break checkpoint/version path determinism.

## 8. Operating Guidance for Scripts
- Use the project hardware slot convention for reproducible runtime comparisons.
- Keep run commands version-explicit and config-snapshot-friendly.
- Treat throughput and reproducibility as coupled requirements, not trade-offs.
