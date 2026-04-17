# Post-Mortems

This document captures failed research paths and the scientific conclusions they established.
The goal is not to hide regressions, but to preserve the negative evidence that shaped the current architecture.

## 1. Foundational Failure Pattern
Across versions, the dominant recurring failure was trying to synthesize missing anatomical information from source contrasts that physically do not encode it.

In practical terms:
- T1w contains strong GM/WM boundaries.
- T2w/FLAIR often attenuate these boundaries.
- No intensity remapping can reliably recover boundaries that are not present in the source signal.

## 2. v10 Post-Mortem: Unsharp Mask Edge Amplification

### Hypothesis
Sharpening source edges would create T1-like structural detail and improve T2w->T1w transfer.

### What was implemented
- 3D unsharp masking in the synthesis path.
- Strong edge emphasis before downstream supervision.

### What failed
- Catastrophic T1w transfer collapse (around 0.18-0.19 Dice in reported runs).
- Segmenter latched onto amplified tumor/CSF boundaries rather than missing cortical tissue boundaries.

### Root cause
Edge amplification is multiplicative on existing gradients. If GM/WM gradients are weak or absent, sharpening cannot hallucinate them. It only magnifies what already exists.

### Scientific conclusion: Information Theory Limit
v10 is the strongest practical demonstration that T2w->T1w boundary recovery is underdetermined in this setting.

## 3. v11 Post-Mortem: Gray Washout from Bezier Warping

### Hypothesis
Continuous nonlinear Bezier remapping would avoid hard quantization artifacts and produce more realistic synthetic contrasts.

### What was implemented
- Random cubic Bezier intensity warps.
- Anisotropic degradation to simulate thick-slice clinical acquisition.

### What failed
- Although some transfer metrics improved versus earlier variants, macro-structural contrast often collapsed toward gray mid-tones.

### Root cause
Bezier curves sampled under broad randomization frequently compressed dynamic range toward central values, reducing tissue-band separation.

### Scientific conclusion
Continuity alone is not enough. Distribution-shaping must explicitly preserve or increase multimodal tissue separability.

## 4. v14 Post-Mortem: Spatial Soft-Quantile Gradients

### Hypothesis
Replacing scalar quantile targets with fully spatial 3D random target fields would increase realism and robustness.

### What was implemented
- Spatially varying soft-quantile targets from coarse random volumes upsampled to full resolution.
- Blended voxel intensities via soft assignments.

### What failed
- Strong regression with gray mush appearance.
- Background masking defects introduced additional artifacts.

### Root cause
Overlapping random spatial targets averaged out effective contrast distinctions. The approach increased stochasticity faster than it increased anatomical signal.

### Scientific conclusion
More spatial randomness is not inherently better. Spatial conditioning must be structured and constrained.

## 5. Formal Strategic Pivot: T1w is All You Need

## 5.1 Why the pivot was necessary
By v10-v14, the project had repeatedly tested major avenues for recovering missing high-frequency anatomical boundaries from lower-frequency contrasts. Failures were consistent with physics and information constraints, not just implementation bugs.

## 5.2 Pivot statement
Primary optimization target shifted to the T1w-source pipeline, where source anatomy already carries the boundary information needed for robust transfer.

## 5.3 What changed after pivot
- Emphasis on contrast diversification while preserving anatomy (v12-v15 line).
- Stronger focus on non-monotonic remapping and consistency regularization.
- Hard background semantics and spatially localized mappings to avoid washout.

## 6. Additional Correctness Post-Mortems (v6 window)
Not all failures were scientific; some were systems defects that temporarily invalidated training conclusions:
- Mask annihilation from non-nearest interpolation on labels.
- Execution-order mismatch between noise and generator guidance path.
- Aliasing artifacts in low-resolution simulation.

These were corrected and documented before interpreting later metrics.

## 7. What We Have Exhausted vs What Remains

### Exhausted or strongly falsified
- Pure Fourier noise as anatomy proxy.
- Edge-only sharpening as boundary recovery mechanism.
- Unconstrained continuous remapping without separability controls.
- Over-randomized spatial target fields.

### Still active and promising
- Structured non-monotonic mappings with explicit spatial locality (v15 lineage).
- Consistency-regularized segmenter training for invariance.
- T1w-first transfer strategy for robust cross-contrast deployment.

## 8. Documentation Policy for Future Failures
Any new regression must be recorded with:
1. Hypothesis and expected mechanism.
2. Reproduction conditions (version, config, checkpoint family).
3. Quantitative failure signature (in-domain, OOD mean, OOD worst).
4. Root cause classification: physics limit, objective mismatch, implementation bug, or evaluation artifact.
5. Go/no-go decision and follow-up action.

This preserves scientific memory and prevents repeated dead-end cycles.

## 10. v17_lpci Startup Failure: AMP Quantile Dtype Mismatch

### Hypothesis
The v17_lpci low-frequency remapping could directly reuse v15 non-monotonic grid chunking in mixed precision generator training.

### Reproduction conditions
- Version: `v17_lpci`
- Task: generator (`t1w` and `t2w`, dual-slot tmux)
- Launch path: `bash scripts/run_generators.sh <slot> v17_lpci <contrast>`
- Precision: `16-mixed`

### Failure signature
- Both generator runs crashed at startup with:
	- `RuntimeError: quantile() input tensor must be either float or double dtype`
	- stack entering `generate_non_monotonic_grid_targets(...)` from `DifferentiableLPCI3D.forward(...)`.

### Root cause
Under AMP, LPCI pyramid tensors were FP16. `torch.nanquantile` in the v15 remapping path requires float/double input, so directly feeding FP16 low-frequency tensors caused a hard runtime failure.

### Tensor-level fix applied
1. In `DifferentiableLPCI3D.forward`, cast low-frequency branch `L2` to FP32 before calling v15 non-monotonic remapping.
2. Cast remapped `L2'` back to the original input dtype after remapping to preserve mixed-precision flow.

### Classification and decision
- Classification: implementation/runtime compatibility bug under mixed precision.
- Decision: mandatory hard restart performed after applying dtype-safe remapping.

## 11. v17_micro_anchor Validation Failure: Half-Precision Peak Mask Overflow

### Hypothesis
The new 1D micro-anchor peak extraction could mask non-max bins using a large negative sentinel and remain stable under mixed precision.

### Reproduction conditions
- Version: `v17_micro_anchor`
- Task: generator validation (`test_gen`, slot 1)
- Launch: `tmux new -s test_gen -d "bash scripts/run_generators.sh 1 v17_micro_anchor t1w"`
- Precision: `16-mixed`

### Failure signature
- Run exited at first epoch start with:
	- `RuntimeError: value cannot be converted to type c10::Half without overflow`
	- source: `generate_micro_anchored_targets(...)` local-max masking sentinel.

### Root cause
The local-maximum masking used a hardcoded `-1e12` sentinel. In FP16, this value overflows representable range and crashes during tensor materialization.

### Tensor-level fix applied
1. Replaced hardcoded sentinel with dtype-safe minimum:
	 - `torch.finfo(h_smooth.dtype).min`
2. Preserved fully vectorized peak extraction path and AMP compatibility.

### Classification and decision
- Classification: mixed-precision numeric stability bug.
- Decision: apply fix and repeat validation protocol (Steps 1-3).

## 12. v18.0 Post-Mortem: Unconstrained Spatial Bezier on Raw Intensities

### Hypothesis
A fully spatially varying cubic Bezier field sampled from unconstrained random control grids could unlock high intra-tissue contrast diversity without quantization artifacts.

### What was implemented
- Coarse control point grids sampled from `U(0,1)` and trilinearly upsampled to dense 3D fields.
- Cubic Bezier mapping applied directly to raw MRI intensities.

### What failed
- Catastrophic macro-contrast loss with pervasive gray-mush appearance.
- Tissue separation collapsed despite nominal nonlinear remapping.

### Root cause 1: Regression to the Mean
Trilinear interpolation of sparse `U(0,1)` control grids drives interior voxels toward the expected central value (`~0.5`). This mean-attractor effect dominates broad tissue regions and produces uniform mid-gray outputs.

### Root cause 2: Missing Histogram Equalization
Applying the Bezier mapping directly to raw intensities bypassed empirical CDF/quantile rank normalization. Because brain tissue often occupies a narrow intensity peak, the Bezier polynomial was effectively evaluated on only a tiny, near-linear local sliver, collapsing macro-contrast.

### Scientific conclusion
Spatial Bezier fields must operate on rank-equalized (quantile/CDF) coordinates and should anchor endpoint semantics explicitly. Unconstrained Bezier-on-raw-intensity mappings are a no-go path.

## 13. v18.1 Post-Mortem: Quantile-Anchored Global Bezier Over-Regularized OOD Structure

### Hypothesis
Anchoring cubic Bezier endpoints on empirical CDF rank space would fix v18.0 gray-mush collapse while retaining strong cross-contrast robustness.

### What was implemented
- Empirical CDF rank mapping (`torch.searchsorted` over sampled quantiles) to move synthesis from raw-intensity to rank coordinates.
- Endpoint anchoring with global inversion coin flip (`P0/P3` fixed to `{0,1}` or `{1,0}`).
- Global cubic Bezier mapping with spatially varying interior controls.

### What worked
- Gray-mush failure mode was largely resolved; contrast dynamic range and tissue separation looked materially better than v18.0.

### What failed
- OOD generalization degraded sharply despite improved visual realism.
- In particular, transfer from T1w to T2w/FLAIR dropped significantly relative to the stronger v15 lineage.

### Root cause
The mapping was still constrained by a single global Bezier polynomial over rank space. Even with spatial interior variation, the operator enforced a continuous monotonic trajectory across tissue bands at each voxel. This prevented independent tissue decoupling/inversion behaviors that v15 could express via non-monotonic chunk remapping. As a result, synthetic supervision became anatomically too regular, and the segmenter overfit canonical anatomical boundaries rather than learning robust OOD-invariant cues.

### Scientific conclusion
Empirical CDF anchoring fixed intensity-collapse pathology but introduced an expressivity bottleneck for OOD robustness. Future versions must preserve rank-space stability while breaking global monotonic coupling between tissue bands.

## 14. v18_2 Post-Mortem: Anchored Extreme Knots Limited Contrast Aggression

### Hypothesis
Moving from global Bezier coupling to a piecewise spatial spline with higher-resolution knot fields would improve OOD invariance while preserving stable intensity semantics.

### What was implemented
- Empirical CDF rank mapping with 100 quantiles and `torch.searchsorted`.
- Piecewise spline interpolation over spatially varying knot targets.
- Fixed global extreme knots with inversion mode:
	- standard: `Y_0=0`, `Y_K=1`
	- inverted: `Y_0=1`, `Y_K=0`

### What worked
- OOD transfer improved materially versus earlier variants.
- Example: T1w->FLAIR rose to approximately `0.396` Dice in observed evaluations.

### What failed
- Synthetic targets remained structurally too tame for worst-case clinical contrast diversity.
- The segmenter still relied on expected intensity spans and under-covered aggressively compressed or globally shifted scans.

### Root cause
Anchoring the spline extremes (`Y_0`, `Y_K`) to `{0,1}` (or `{1,0}` under inversion) hard-constrained dynamic range endpoints. This prevented the generator from expressing collapsed-range mappings and stronger global contrast shifts, reducing the space of realistic low-contrast and strongly shifted domains.

### Scientific conclusion
Rank-space spline control improved robustness, but endpoint anchoring imposed an expressivity ceiling. v18_3 should remove endpoint anchoring entirely and use fully unanchored free-knot spatial targets.

## 15. v18_3 Post-Mortem: Free-Knot Splines Preserved Boundaries Too Well

### Hypothesis
Removing endpoint anchors and using fully unanchored spatial free-knot targets would maximize contrast-domain diversity and improve OOD robustness.

### What was implemented
- Empirical CDF rank mapping with 100 quantiles and vectorized `torch.searchsorted` rank assignment.
- Fully unanchored `K=8` spline knot fields sampled independently over space.
- AMP-safe vectorized piecewise interpolation with strict background masking.

### What worked
- v18_3 significantly improved OOD mean and stabilized broad transfer behavior relative to prior anchored variants.

### What failed
- Healthy anatomical boundaries were preserved too perfectly across adjacent tissue bands.
- Synthetic supervision remained overly boundary-faithful, especially for contrasts where boundaries are physically attenuated.

### Root cause
Assigning distinct targets to every neighboring tissue quantile mathematically preserves inter-quantile gradients. Even with unanchored knots, if each adjacent rank interval maps to a different target trajectory, boundaries survive by construction.

### Core insight

The direct conclusion is that preserving a rank-space spline alone is not enough; the generator must also vary how much raw identity survives in each spatial region. `v18_7` does that by stochastic leakage instead of a deterministic continuous remap.

## 16. v18_7 Design Rationale: Dynamic Stochastic Identity Leakage

### Motivation
`v18_6` proved that full synthetic augmentation can substantially stabilize OOD behavior, but the in-domain T1w score still degraded because the segmenter no longer saw enough pristine anatomy.

### Strategy
`v18_7` introduces a controlled leakage path that mixes the raw image back into the synthetic target while keeping the aggressive remapping path intact.

Mechanically:
- Randomize the number of quantile bins per batch so the target generator cannot settle on one fixed partitioning scheme.
- Randomize the spatial alpha grid from `1^3` up to `8^3` so the identity leak is spatially heterogeneous rather than globally smooth.
- Blend the raw image directly into the generated target with a dense alpha field, forcing the segmenter to handle raw biological boundaries and heavily distorted synthetic features within the same training sample.

### Why this should work
The goal is not to remove the OOD robustness discovered in `v18_6`; it is to recover some of the in-domain precision that was lost when the model was exposed to fully synthetic supervision at probability 1.0.

The stochastic leak keeps the network from learning an optimization shortcut that simply copies identity structure, because the leak strength and spatial structure change every batch. At the same time, it prevents the synthetic objective from becoming so disconnected from raw anatomy that the segmenter forgets crisp biological boundaries entirely.

### Additional bottlenecking
`v18_7` also reduces the generator footprint by cutting `base_filters` to 8. That enforces a stricter information bottleneck and reduces forward/backward cost, which is important because the new stochastic target construction adds more runtime work than the earlier deterministic mapping.

## 17. v18_5 Post-Mortem: Spline-Family CDF Stretching Produced Metallic Tissue Artifacts

### Hypothesis
Spline-based rank-space mappings (v18_1 through v18_5) could preserve stochastic variation while allowing strong contrast-domain remapping without introducing quantization artifacts.

### What was implemented
- `v18_1` through `v18_5` all mapped empirical CDF/rank coordinates through continuous interpolation families (Bezier/spline variants), with progressively stronger boundary suppression and optional heavy guidance smoothing.

### What failed
- Across the spline family, synthetic volumes developed non-biological "metallic" or "plastic" reflection-like shading.
- Artifact severity increased when pushing aggressive inversions or broadened dynamic range manipulations.

### Root cause
Continuous curve interpolation over CDF ranks imposes smooth gradient structure on regions where the original tissue distribution is naturally tight and noisy. In MRI tissue clusters, local variation is dominated by residual stochastic texture, not wide uniform ramps. Rank-space spline stretching converts these dense noisy clusters into artificially uniform gradients, yielding the observed metallic/plastic shading.

### Scientific conclusion
The failure is structural to the spline-over-rank family, not a tuning issue in knot count, anchors, coalescence, or blur strength. Any continuously interpolated CDF-rank remapper tends to erase biologically plausible micro-texture when forced to perform aggressive remapping.

### Pivot decision (v18_6)
Return to discontinuous piece-wise linear mapping on raw intensities with texture-preserving residuals:
- Partition intensities into discrete quantile chunks.
- Shift chunk base color with independent random chunk target means.
- Preserve each voxel's intra-chunk residual noise via local linear scaling around the chunk lower edge.

This gives aggressive non-monotonic boundary destruction while preserving biologically plausible local texture, avoiding CDF-stretch metallic artifacts.
To generalize to boundary-poor contrasts (for example T2w/FLAIR), the synthetic target generator must intentionally destroy healthy tissue boundaries. Adjacent quantile bands must be forced to collapse onto shared targets so their separating gradients are annihilated, explicitly simulating topological tissue merging.

### Scientific conclusion
v18_4 must add explicit quantile-band coalescence rather than only free-knot randomness. Without forced adjacent-band collapse, edge overfitting remains structurally incentivized.

## 16. v18_4 Post-Mortem: Knot Coalescence Over-Destroyed Useful Structure

### Hypothesis
Randomly coalescing adjacent spline knots would annihilate healthy boundary shortcuts and improve OOD robustness on boundary-poor contrasts.

### What was implemented
- Vectorized knot coalescence over `K=8` spatial spline channels using keep masks and `torch.cummax` forward-filled indices.
- Coalesced knot gathering with pure tensor indexing (`torch.gather`) and AMP-safe interpolation in rank space.

### What worked
- Slight FLAIR transfer improvement versus v18_3.
- No startup stability regressions; operator remained fully vectorized and throughput-safe.

### What failed
- OOD mean regressed from approximately `0.543` (v18_3) to `0.534` (v18_4).
- T2w transfer also regressed, indicating loss of clinically useful inter-tissue structure.

### Root cause
Boundary destruction was likely too aggressive. Coalescing adjacent quantile bands removed not only shortcut edges but also structurally informative mid/high-frequency cues needed for robust cross-contrast generalization.

### Pivot
v18_5 reverts target generation to `v18_3` unanchored splines, but introduces heavy low-pass filtering on the guidance map. This keeps low-frequency intensity supervision while forcing the generator to recover high-frequency structural edges from the source image instead of copying them directly from guidance.

### Scientific conclusion
The best operating point is controlled edge suppression, not full topological collapse. Heavy guidance low-pass filtering should decouple contrast cues from structural boundary shortcuts while preserving useful macro-structure.

## 9. v16_bigaug Restart Post-Mortem: Throughput SLO Violation on First Launch

### Hypothesis
Implementing Zhang-style deep stacked augmentations with all nine transforms and fused spatial resampling would preserve robustness while remaining below the segmenter throughput SLO.

### Reproduction conditions
- Version: `v16_bigaug`
- Task: segmenter baseline (`use_generator=false`)
- Launch pattern: dual tmux sessions via `bash scripts/run_segmenters.sh 1 v16_bigaug t1w` and `bash scripts/run_segmenters.sh 2 v16_bigaug t2w`
- Initial default batch size: 4

### Failure signature
- First training launch exceeded SLO during early epoch timing (approximately 33s/epoch observed in startup epoch progress), above the `<14s/epoch` requirement.

### Root cause
The initial BigAug implementation executed several dense transforms over the full batch regardless of per-transform Bernoulli activation masks. This created avoidable compute pressure:
- appearance transforms were computed for inactive samples,
- elastic field synthesis was computed even when deformation mask was inactive,
- spatial warp executed over full batch rather than active subset.

### Tensor-level optimizations applied before restart
1. Converted appearance transforms to active-subset execution (`x[mask]`) so compute scales with expected active fraction rather than full batch.
2. Added deformation short-circuit: skip elastic field generation entirely when no samples activate deformation.
3. Switched fused spatial warp to operate on active spatial subset only, then scatter results back.
4. Reduced elastic field generation resolution from factor 4 to factor 8 before trilinear upsampling, preserving smooth deformation while reducing kernel workload.
5. Enabled compile for the BigAug augmentation module (`torch.compile(..., mode="reduce-overhead")`) under segmenter compile mode.
6. Set `v16_bigaug` launcher default batch size to 8 (unless explicitly overridden) to reduce per-epoch step count and improve wall-clock throughput.

### Classification and decision
- Classification: implementation/performance engineering bottleneck (not objective mismatch).
- Decision: hard restart mandated and executed under strict policy with optimized BigAug path.

### Second restart note (same version window)
- A subsequent relaunch exposed severe startup throughput collapse tied to `torch.compile` graph partitioning/cudagraph skips in the BigAug module due dynamic masked mutation paths.
- Remediation applied:
	1. Removed compile wrapping for BigAug augmentation path.
	2. Reworked intensity stack to run at half spatial resolution and upsample back, preserving transform contract with substantially lower blur compute.
- Decision: second hard restart required under strict SLO policy.

## 18. Consolidated Post-Mortem: The Spline "Metallic Reflection" Artifact (v18_1 to v18_5)

### Failure statement
Continuous CDF/rank spline families from `v18_1` through `v18_5` repeatedly generated non-biological metallic/plastic tissue appearance under aggressive remapping.

### Mechanism
- Brain tissue intensity clusters are naturally tight and stochastic (approximately Gaussian-like local bands with micro-texture).
- Continuous spline stretching redistributes these tight clusters over smooth ramps.
- The remapper imposes artificial low-frequency shading gradients that resemble 3D lighting/reflection, not MRI biology.

### Practical impact
- Biological texture cues are destroyed.
- The segmenter receives unrealistic supervision and overfits synthetic shading artifacts.

### Decision
The spline-over-CDF family is treated as structurally unsafe for aggressive remapping in this project. `v18_6` replaced it with discontinuous texture-preserving chunk remapping to preserve intra-band noise while still enabling strong contrast inversion.

## 19. Post-Mortem: Generative TTA Domain-Shift Paradox

### Failure statement
Synth-only / generative TTA evaluation failed because the generator is explicitly trained on a T1w-source domain and is not domain-agnostic.

### What was attempted
- At test time, target-domain scans (for example FLAIR/T2w) were fed into the T1w-trained generator to synthesize additional TTA variants.

### What failed
- Catastrophic OOD feature explosion in generator activations.
- Generated volumes became garbage-like and degraded downstream segmentation.

### Root cause
Generative TTA assumes the synthesis model remains stable for the inference-domain input manifold. Our generator is intentionally specialized to T1w source statistics. Feeding non-T1w contrasts at evaluation violates that manifold assumption and induces distributional breakdown.

### Scientific conclusion
Generative TTA is only valid when the generator itself is domain-agnostic or explicitly trained for the target-domain manifold. In this project, zero-shot evaluation must avoid generator-driven TTA on non-T1w inputs.

## 20. v20 Post-Mortem: Partial SynthSeg White-Noise Collapse

### Hypothesis
Applying SynthSeg-style per-class Gaussian sampling only on sparse tumor labels (1/2/3) would provide a fair ablation baseline without requiring dense healthy-brain labels.

### What was implemented
- Tumor-only class sampling with independent Gaussian draws.
- Healthy/background region largely preserved from the source scan.

### What failed
- Segmenter stability and OOD transfer collapsed.
- Synthetic tumor regions became texture-incoherent and statistically disconnected from surrounding anatomy.

### Root cause
The augmentation path injected voxel-independent Gaussian noise (white noise) inside class masks. White noise has no local spatial correlation. 3D CNNs rely on spatially coherent local neighborhoods to learn useful anatomy-grounded filters; replacing masked regions with uncorrelated noise destroys that signal and encourages brittle overfitting.

### Scientific conclusion
Pure white-noise class replacement is not a valid MRI texture model for sparse-label synthesis. Any SynthSeg-style sparse-label baseline must preserve spatial correlation structure.

## 21. v20_1 Hotfix: Spatially Correlated Noise via Separable Gaussian Filtering

### Fix
Convert white noise into spatially correlated noise before mapping it to tumor masks:
1. Sample raw Gaussian noise tensor $Z \sim \mathcal{N}(0,1)$.
2. Apply separable 1D Gaussian blur along depth, height, and width.
3. Normalize variance and then apply class-wise affine mapping inside masks.

### Why this works
- The separable blur restores local correlation and biological-like continuity.
- It reproduces the key resolution-smoothing behavior expected by SynthSeg-style intensity synthesis.
- It is computationally efficient in 3D because the blur is separable, preserving an $O(3N)$-style cost profile instead of dense-kernel $O(N^3)$ scaling.

### Outcome
The hotfix removed the pathological white-noise behavior and produced a trainable partial-SynthSeg baseline with realistic local texture continuity.

## 22. v21: Online Sparse SynthSeg Baseline — Scientific Redesign
**Date:** 2026-04-13
**Status:** Active — empirical lower-bound campaign

### Prior offline attempt (superseded)
An earlier v21 attempt applied the unmodified `BrainGenerator` from the `SynthSeg` codebase directly onto sparse BraTS labels to generate a static pre-synthesized dataset. That baseline was **scientifically unfair**: a fixed finite dataset cannot match the infinite variance of our online augmentation methods, meaning any observed performance gap could be attributed to dataset diversity rather than algorithmic quality.

### Redesign: online GPU augmentation
**Issue with the dense-label paradigm:**
SynthSeg expects a fully segmented anatomical label map (GM, WM, CSF, ventricles, etc.) to fit per-class GMMs that reflect real tissue statistics. With only sparse BraTS labels, the healthy brain has no dedicated anatomical labels — we cannot synthesise it. We therefore preserve the original brain intensities for label 0 and apply GMM sampling only to the tumor subregions (labels 1/2/3).

**What changed from the initial design:**
An earlier v21 draft applied GMM sampling to ALL four classes (including label 0 = healthy brain), which caused the entire brain to disappear into homogeneous noise. This made training trivially broken — the segmenter had no anatomical signal to learn from, producing near-zero training Dice rather than the scientifically interesting result of a trainable-but-OOD-failing model. The corrected design preserves the original scan for the healthy brain and applies hard GMM replacement only to tumor subregions.

**Scientific fix for fairness:**
`v21` implements SynthSeg-style GMM + spatial-blur as an **online GPU augmentation** (`SparseSynthSegAugmentation3D`), giving the baseline identical infinite-variance data as `v19`/`v20_1`. The "floating tumor" failure mode is that each training batch randomises tumor intensities entirely independently of the surrounding brain context, so the segmenter cannot learn anatomy-grounded OOD features.

**Key distinction from v20:**
v20 (`PartialSynthSegAugmentation3D`) uses a soft spatial PSF blend at tumor boundaries. v21 uses a hard pixel-level replacement with no anatomical smoothing — sharper, more physically implausible tumor boundaries, and a harder learning problem.

**Expected outcome:**
- In-domain training Dice: moderate (model learns statistical anomalies within a realistic brain context).
- OOD Dice: poor (random tumor intensities do not encode the anatomy-contrast relationships present in real scans).
- v21 becomes the rigorous empirical lower bound over v20.
