# Knowledge Transfer: Architectural Evolution (v7 to v15)

## The Fundamental Problem
The core challenge in unsupervised single-source MRI contrast synthesis is the "Physics Asymmetry Gap". Synthesizing a low-frequency image (T2w/FLAIR) from a high-frequency source (T1w) is mathematically easier than the reverse. T1w contains distinct Gray Matter/White Matter (GM/WM) boundaries; T2w lacks them due to relaxation time physics.

## Version-by-Version Breakdown

* **v7 (Fourier Amplitude Randomization):** Attempted to inject high-frequency edges into T2w by randomizing Fourier amplitudes. Result: Excellent for FLAIR generalization, but failed for T1w (0.19 Dice) because Fourier noise is unstructured static, not biological anatomy.
* **v8 (Spatially-Varying Grid Chunking):** Implemented local quantile thresholds on a 4x4x4 grid with trilinear interpolation to create macro-regional contrast gradients. Retained 30% Fourier. Result: SOTA for single-source macro-structural synthesis. Solved T2w -> FLAIR (0.72 Dice). The official baseline.
* **v9 (Procedural Micro-Texture):** Injected 3D Perlin/Fractal noise to hallucinate micro-edges. Result: Regression (0.40 OOD). The Gaussian blur smeared the noise, and the segmenter learned to ignore static rather than parse real anatomy.
* **v10 (Anatomical Edge Sharpening):** Used a 3D Unsharp Mask to amplify existing T2w boundaries. Result: Catastrophic collapse on T1w (0.18 Dice). *Scientific Takeaway:* Amplified tumor/CSF edges instead of GM/WM because GM/WM edges physically don't exist in T2w. Proved the **Information Theory Limit** of single-source generation.
* **THE STRATEGIC PIVOT ("T1w is All You Need"):** Acknowledged that T2w -> T1w is mathematically impossible without external priors. Shifted primary focus to optimizing the T1w -> Any pipeline.
* **v11 (Non-Linear Bezier Warping & Anisotropic Degradation):** Replaced discrete chunking with continuous, randomized cubic Bezier splines. Added Z-axis downsampling to simulate clinical slice thickness. Result: Massive jump in T2w->T1w generalization (0.40) due to continuous mapping, but caused "gray washout" of macro-structures because Bezier curves compress contrast to the mean.
* **v12 (Multi-Peak GMM Histogram Matching):** Used CDF matching against a random Multi-Peak GMM to fix the gray washout and force vivid contrast bands. Purged Fourier noise. Result: Restored sharp boundaries (T2w->T1w hit 0.49). *Flaw:* CDF matching is strictly monotonic, preventing contrast inversions (dark could never become bright).
* **v13 (Soft-Quantile Shuffling & Consistency Regularization):** Replaced monotonic CDF matching with temperature-scaled soft-assignments to random, independent targets (allowing contrast inversion). Added a Jensen-Shannon/KL Consistency Loss (via batch concatenation) to anchor the segmenter to real T1w anatomy. Result: Phenomenal breakthrough. T2w in-domain hit 0.814, OOD on T1w hit 0.525.
* **v14 (Spatially-Varying Soft-Quantiles):** Attempted extreme variability by mapping tissues to 3D gradients instead of scalars. Result: Failed. Overlapping 3D gradients averaged out into gray mush, and background masking was bugged.
* **v15 (Non-Monotonic Grid Chunking):** Major hybrid milestone. Combines the spatial sharpness of v8's grids with the contrast-inverting non-monotonicity of v13, plus explicit background masking.

## Loss Landscape Contradictions (v18_6 Lesson)

When a target or guidance map is intentionally blurred, the loss stack must be made logically consistent with that decision.

What failed in `v18_5`:
- Guidance was heavily low-pass filtered.
- At the same time, high-frequency L1 guidance penalties (for example `guidance_sharp`) still forced the model to match crisp details against a blurred target.

Why this is contradictory:
- The blur objective says: do not trust high-frequency detail in guidance.
- The sharp-L1 objective says: exactly reproduce high-frequency detail from guidance.
- The optimizer receives mutually incompatible gradients and punishes anatomically correct sharp boundaries recovered from the raw source image.

v18_6 correction:
- Keep blurred guidance to remove shortcut edges.
- Remove/disable high-frequency L1 penalties against that blurred guidance.
- Let dominant edge-aware supervision and source-image structure drive crisp anatomy recovery.

Transferable rule:
- If you intentionally low-pass a supervision signal, do not keep high-frequency reconstruction penalties tied to that same signal.

## The Regularization Power of Total Synthetic Starvation

Setting the segmenter training augmentation probability to 1.0 changes the optimization problem in a useful way. It trades a small amount of peak in-domain performance, roughly from the ~0.70 range down to ~0.62, for much stronger out-of-distribution stability and cross-contrast invariance.

The mechanism is simple: once the model is denied raw source-domain views during training, it can no longer lean on domain-specific intensity bands as shortcuts. It is forced to learn pure topological structure and anatomy-consistent shape priors, which is exactly what improves transfer when contrast changes at test time.

This is why the `v18_6` augmentation-probability-1.0 setting matters. It does not make T2w intrinsically richer than T1w, but it does show that total synthetic starvation is a powerful regularizer for segmenter robustness.