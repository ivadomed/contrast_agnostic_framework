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
