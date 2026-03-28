# Version History

This document summarizes the project evolution from v1 to v15 in a consistent scientific format:
Hypothesis -> Implementation -> Result -> Scientific Takeaway.

## v1
- Hypothesis: A basic generator-guided training loop can improve cross-contrast robustness over single-contrast supervision.
- Implementation: Initial generator + segmenter workflow with early histogram/guidance ideas.
- Result: Strong in-domain from T1w source (in-domain Dice 0.7486), weak T2w->T1w transfer (t1w Dice 0.1059 when source is T2w).
- Scientific Takeaway: Core asymmetry was visible early: low-frequency contrasts do not contain enough intrinsic boundary information for high-frequency reconstruction.

## v2
- Hypothesis: Better baseline and generator orchestration will improve reliability across source contrasts.
- Implementation: Baseline references and v2 generator variants evaluated under a unified scoring layout.
- Result: Baselines remain source-dependent; generator variants improve some OOD paths but do not remove asymmetry.
- Scientific Takeaway: Contrast source identity dominates outcome quality unless augmentation/guidance is explicitly designed to attack missing-boundary physics.

## v3
- Hypothesis: Fully artificial contrast training can reduce source-specific overfitting.
- Implementation: Added fully artificial training families alongside baseline and generator families.
- Result: Fully artificial T1w branch achieved broad OOD strength (ood_mean up to 0.6608), but T2w branch remained weaker and volatile.
- Scientific Takeaway: Synthetic diversification helps robustness, but does not fully recover information that is absent in the source contrast.

## v4
- Hypothesis: Reproducibility engineering will improve experiment credibility and enable fair version-to-version comparisons.
- Implementation: Migration to Hydra + PyTorch Lightning, seeded execution, standardized logging/checkpointing/resume, reproducibility metadata, and core operator tests.
- Result: Stable and auditable experiment framework; reduced script-level variance.
- Scientific Takeaway: Systems discipline is a prerequisite for valid scientific claims in fast-iteration research.

## v5
- Hypothesis: Throughput optimization can accelerate iteration without changing the core scientific objective.
- Implementation: Vectorized histogram/augmentation paths, separable 1D Gaussian convolutions, compile-boundary cleanup, reduced logging overhead.
- Result: Significant speed gains with mixed metric stability periods.
- Scientific Takeaway: Speed-first phases can expose hidden fragility; performance wins must be paired with correctness monitoring.

## v6
- Hypothesis: Consolidating speed work and fixing augmentation correctness defects will restore reliable learning dynamics.
- Implementation: Fixed mask interpolation semantics (nearest for labels), corrected operation order, repaired low-res aliasing paths, hardened RNG and augmentation return contracts, extended profiling-based optimizations.
- Result: Correctness recovered; ensemble studies showed only small non-monotonic improvements.
- Scientific Takeaway: Many regressions were pipeline/correctness issues, not just model-capacity limits.

## v7
- Hypothesis: Fourier amplitude randomization can inject missing high-frequency detail for low-frequency source contrasts.
- Implementation: FFT-domain high-frequency amplitude perturbation with phase preservation; version-gated integration in synthesis and segmenter paths.
- Result: Improved some OOD behavior (notably FLAIR), but severe T2w->T1w weakness persisted.
- Scientific Takeaway: Unstructured Fourier texture is not a substitute for biologically meaningful anatomical boundaries.

## v8
- Hypothesis: Spatially varying chunk-based mappings can better model macro-regional contrast behavior than global transforms.
- Implementation: Local quantile thresholds on coarse 3D grids with trilinear interpolation; partial Fourier retention.
- Result: Major gain in macro-structural transfer; became official baseline lineage.
- Scientific Takeaway: Spatial locality is critical; global intensity transforms alone underfit anatomical heterogeneity.

## v9
- Hypothesis: Procedural micro-texture (Perlin/fractal style) can recover missing fine details.
- Implementation: Injected synthetic 3D micro-noise patterns into guidance path.
- Result: Regression in OOD robustness.
- Scientific Takeaway: Noise that is not anatomically grounded is learned as nuisance and can be ignored by the segmenter.

## v10
- Hypothesis: Sharpening anatomical edges in source images can recover T1-like boundaries.
- Implementation: 3D unsharp masking to amplify existing boundaries.
- Result: Catastrophic T1w transfer collapse (around 0.18-0.19 Dice in key paths).
- Scientific Takeaway: Information Theory Limit established. Amplifying existing T2w edges boosts tumor/CSF boundaries, not missing GM/WM structure.

## Strategic Pivot: T1w is All You Need
- Hypothesis: Because T2w->T1w synthesis is fundamentally underdetermined, the highest-value path is maximizing T1w->any robustness.
- Implementation: Refocused generator/segmenter design goals toward robust contrast diversification from T1w-rich boundary information.
- Result: Subsequent versions prioritized invertible/non-monotonic and spatially controlled mappings with consistency constraints.
- Scientific Takeaway: Scientific progress required changing objective framing, not only tuning hyperparameters.

## v11
- Hypothesis: Continuous nonlinear remapping (Bezier) plus anisotropic degradation improves realism and transfer.
- Implementation: Cubic Bezier intensity warps + Z-axis downsampling simulation.
- Result: Large T2w->T1w gain versus older designs, but widespread gray washout.
- Scientific Takeaway: Continuity improved stability, but unconstrained nonlinear compression collapsed dynamic range.

## v12
- Hypothesis: Multi-peak GMM histogram targets can restore vivid multimodal tissue separation after v11 washout.
- Implementation: Random GMM target CDF matching, Fourier purge for v12, background-preserving histogram remap.
- Result: Better boundary separability and stronger transfer (T2w->T1w around 0.49 in key runs).
- Scientific Takeaway: Multi-modal targets help, but strict CDF monotonicity blocks full contrast inversion.

## v13
- Hypothesis: Non-monotonic soft-quantile shuffling plus consistency regularization can combine diversity and anatomical stability.
- Implementation: Temperature-scaled soft assignments to independent random targets, plus KL-style raw-vs-synth consistency using batch-concatenated dual-pass training.
- Result: Breakthrough robustness (e.g., T2w in-domain ~0.814, strong OOD gains including T1w ~0.525 in key reports).
- Scientific Takeaway: Non-monotonic mappings are essential; consistency regularization prevents synthetic-style drift.

## v14
- Hypothesis: Spatially varying soft-quantile targets can further improve diversity by replacing scalar targets with 3D fields.
- Implementation: Coarse 3D random target volumes upsampled and mixed by soft quantile weights.
- Result: Failed due to gray mush and masking issues.
- Scientific Takeaway: Excessive spatial stochasticity can destroy tissue separability when target fields overlap and average out.

## v15
- Hypothesis: Combine v8 spatial sharpness with v13 non-monotonicity, while enforcing strict background semantics.
- Implementation: Non-monotonic grid chunking with dense interpolated thresholds, unsorted chunk targets, and explicit tissue/background masking; integrated with inherited anisotropic degradation and consistency regularization gates.
- Result: Current SOTA candidate with strong in-domain and balanced OOD for T1w source branch (ens1: in-domain 0.7060, OOD mean 0.6257).
- Scientific Takeaway: Best-performing design is a hybrid: spatially structured mapping + non-monotonic inversion + hard background constraints.

## Summary Across Versions
- v1-v3 established baseline asymmetry and early synthetic training behavior.
- v4-v6 established reproducibility/performance/correctness foundations.
- v7-v10 experimentally falsified several edge-hallucination strategies.
- v11-v15 converged toward the current hybrid architecture, with v15 as the leading candidate pending future external validation.
