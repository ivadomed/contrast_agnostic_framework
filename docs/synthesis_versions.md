# Synthesis version log

Naming: `v<major>_<minor>`. Major increments on architectural changes; minor on hyperparameter/sampling variants.  
LHC (Sobol quasi-random parameter sampling) is a generation flag, not part of the version name.

---

## Architecture note — guidance-map-only generation

Starting from the v26 experiments, synthetic volumes are generated from the **guidance map alone** — the target generator's intensity-remap output — without passing through the U-Net synthesizer.

Script: `scripts/generate_synthetic_guidance.py`  
Why: (1) orders of magnitude faster (no forward pass), (2) avoids U-Net hallucinations at extreme contrasts, (3) allows rapid iteration on the target-generator design without retraining.

The guidance map is a piecewise-affine-remapped T1w volume, blurred with a random Gaussian (σ list `[0.0, 0.0, 0.0, 0.3, 0.5, 0.8]`) and optionally downsampled (zoom ∈ [0.4, 1.0]). It is used directly as the synthetic MRI volume for feature extraction and manifold analysis.

The U-Net is still used during training (guidance map → synthesizer → reconstructed contrast), but the analysis pipeline evaluates guidance maps directly.

---

## v19 — Chunk remap baseline
Foreground split into K=8 uniform quantile chunks; each chunk gets an independent (mu, alpha) affine remap.  
U-Net is trained end-to-end on top of the guidance map.

## v22_1, v22_2 — LHC sampling
Same architecture as v19. `_1` = best recall / manifold alignment. `_2` = highest OOD diversity (Vendi), lower recall.

## v23_1 — Random K
K drawn randomly each forward pass from {2, 3, 4, 6, 8, 12, 16} (log-spaced).  
With K=2 the probability of T2w-like contrast (all WM chunks low, all GM chunks high) rises to ~16%.

## v23_2 — Foreground-quantile chunk boundaries (V24FgQuantile under the hood)
Chunk edges placed at foreground quantile percentiles instead of uniform linspace.  
Ensures each chunk covers equal brain-tissue mass, allowing better separation of CSF/GM/WM with small K.

## v23_3 — Random blur + random resolution
Base: V23RandomChunkTargetGenerator (same as v23_1).  
Added per-variant random Gaussian blur σ ~ U(0.3, 3.0) and optional resolution downsampling zoom ~ U(0.4, 1.0) before the upsample-blur-downsample pass.  
Goal: add diversity to the guidance maps without any contrast-specific bias.

## v23_4 — Log-skewed blur (v23_3 + better blur distribution)
Base: V23RandomChunkTargetGenerator (same as v23_3).  
Blur sigma distribution changed from uniform U(0.3, 3.0) to: 30% no blur (sigma=0), 70% log-uniform over [0.3, 3.0]. The log-uniform concentrates mass at low sigma (~50% below 0.95) while still allowing heavy blur occasionally.  
Random resolution zoom unchanged: U(0.4, 1.0). Uses v23_3 checkpoint.

## v25_1 — Ellipsoidal blob modulator on v23_3
Base: V23RandomChunkTargetGenerator.  
A base guidance map is generated for the whole volume (same as v23_3). Then 0–3 random ellipsoidal blobs are stamped on top: each blob has a random center in the brain foreground, anisotropic radii drawn from U(15, 80) voxels per axis, and a hard binary boundary. Each blob region is filled with an independent V23RandomChunk guidance map call (different K, mu, alpha).  
Effect: localized regions with distinct intensity remapping, preserving boundaries, without any anatomical prior (no labels, no atlas).  
Also inherits random blur σ ~ U(0.3, 3.0) and random resolution zoom ~ U(0.4, 1.0) from v23_3.

## v25_2 — v25_1 with v23_4 blur schedule
Same spatial blob modulator as v25_1, but the blur schedule switches to the v23_4 log-skewed blur distribution: 30% no blur, 70% log-uniform over [0.3, 3.0].  
Random resolution zoom remains U(0.4, 1.0). Uses the same v23_3 checkpoint family as v25_1.

---

## v26 family — EM spatial parcellation

All v26 variants replace global intensity binning with 1D K-means on T1w foreground + optional Voronoi spatial sub-parcellation. They use the v23_3 checkpoint and guidance-map-only generation.  
Blur: list `[0.0, 0.0, 0.0, 0.3, 0.5, 0.8]` (weighted toward no/low blur). Resolution zoom: U(0.4, 1.0).

### v26_1
**Generator:** `V26EMParcellationChunkTargetGenerator`  
C ∈ {2–6} K-means intensity classes; each class Voronoi-split into S ∈ {2–10} spatial sub-regions. Each sub-region gets independent (mu, alpha) affine remap.  
Goal: fully label-free spatial parcellation that disentangles same-intensity tissues by location.

### v26_2
**Generator:** `V26_2EMParcellationTargetGenerator`  
30% of images skip parcellation entirely (global affine fallback). Per-class 40% chance to skip Voronoi sub-division (S=1). Fixed discrete blur targets {0.0, 0.5, 1.0, 1.5}.

### v26_3
**Generator:** `V26_3EMParcellationTargetGenerator`  
Skip-parcellation prob reduced to 10% (vs 30% in v26_2). Sub-parc skip unchanged at 40%. Added per-sample **min-max normalization** inside brain mask (replaces hard clamp to [0,1]) — preserves the full dynamic range of the affine remap without saturation.

### v26_4 — **best baseline**
**Generator:** `V26_4EMParcellationTargetGenerator` (inherits v26_3 with `pass`)  
Identical to v26_3. Min-max normalization was the v26_3 change; v26_4 made that the canonical baseline.  
**Manifold results (regional_hist_64, PCA@60):** 41/42 IND groups, recall 0.985, density 0.688, Vendi 2.775, OOD norm 16.91.

---

## v26_5 — Polarity flip (class-level shared mu)
**Generator:** `V26_5PolarizedTargetGenerator`  
50% chance per image: sort K-means classes by T1w centroid, assign mu values in *descending* order (darkest tissue → highest output brightness). Sub-regions within each class share the class-level mu; only alpha varies spatially.  
**Result:** Hurt coverage badly (32/42 IND groups). Sharing class-level mu reduced within-region spread, collapsing T2w group coverage (7→2). Polarity flip at class level is the wrong granularity.

## v26_6 — **current best** — Signed alpha
**Generator:** `V26_6SignedAlphaTargetGenerator`  
Same as v26_4 but alpha can be negative: `sign ~ Bernoulli(0.5); alpha = sign × U(0.5, 2.0)`.  
Negative alpha inverts the local intensity ordering within each sub-region (bright T1w voxels → dark output and vice versa), expanding the accessible histogram shapes without breaking the global parcellation structure.  
**Manifold results (regional_hist_64, PCA@60):** **42/42 IND groups** (first method to achieve perfect coverage), recall 0.989, density 0.652, Vendi 2.802, OOD norm 17.16.  
Gained T2w group 8; bold and dwi recall reach 1.000.

## v26_7 — Flat regions (constant per sub-region)
**Generator:** `V26_7FlatRegionTargetGenerator`  
Same K-means + Voronoi parcellation as v26_4 but replaces the affine remap with a pure constant: `y[mask] = mu`. No alpha term.  
Conceptually equivalent to SynthSeg-modeB applied with our own parcellation instead of sklearn GMM.  
**Manifold results:** 27/42 IND groups, recall 0.977, **Vendi 3.686**, OOD norm 14.33. High Vendi comes from degenerate (delta-spike) regional histograms that land in extreme PCA corners — not meaningful diversity for training. T1w coverage collapsed (2/8 groups) because smooth within-region distributions are needed to match real T1w.

## v26_8 — Signed alpha + 50% global inversion
**Generator:** `V26_8GlobalInversionTargetGenerator`  
v26_6 parcellation, then with 50% probability flip `y → 1-y` within the brain mask.  
**Manifold results:** 42/42 IND, recall 0.971 ↓, density 0.616 ↓, Vendi 2.754 ↓. Global inversion after per-region remap adds no new information (the remap already explores all polarities). Recall slightly hurt.

## v26_9 — Signed alpha + log-uniform gamma
**Generator:** `V26_9GammaToneTargetGenerator`  
v26_6 parcellation, then apply `y → y^γ` where `γ ~ LogUniform(0.25, 4.0)`.  
**Manifold results:** 41/42 IND, recall 0.990, Vendi **2.816** — highest Vendi within the affine-remap family while maintaining near-full coverage. Lost one group vs v26_6. Marginal gain only.

## v26_10 — Signed alpha + additive fractal noise
**Generator:** `V26_10FractalNoiseTargetGenerator`  
v26_6 parcellation, then add `σ × noise_fractal` where `σ ~ U(0.05, 0.25)`.  
**Manifold results:** 42/42 IND, recall 0.970 ↓, Vendi 2.765. Fractal noise adds spatial texture variation but hurts recall (noise blurs the precise histogram shape needed for IND classification). HOG-space coverage improvement was not significant.

## v26_11 — Large-C signed alpha (no Voronoi)
**Generator:** `V26_11LargeKTargetGenerator`  
C ∈ {8, 10, 12, 16} K-means classes, signed alpha per class, no Voronoi sub-parcellation.  
**Manifold results:** 41/42 IND, recall 0.972, Vendi 2.768. More classes do not increase Vendi — larger C subdivides T1w intensity finer but the anatomical region histograms stay in the same PCA neighborhood.

## v26_12 — Stratified mu sampling
**Generator:** `V26_12StratifiedMuTargetGenerator`  
After K-means, divide [0,1] into C equal bins and draw one mu per bin, then randomly shuffle assignments to classes. Guarantees every image has one class in each brightness band.  
**Manifold results:** 41/42 IND, recall 0.967, Vendi **2.745** — *lower* than v26_4. Forced stratification makes all images more similar (each always has dark/medium/bright zones), reducing diversity vs independent random draws.

## v26_13 — Large-C + stratified mu
**Generator:** `V26_13LargeKStratifiedTargetGenerator`  
Combines v26_11 (large C) and v26_12 (stratified mu), no Voronoi.  
**Manifold results:** 40/42 IND, recall 0.964, Vendi 2.765. Combination is strictly worse than either alone.

## v26_14 — Mixed flat + affine (40% flat per class)
**Generator:** `V26_14MixedFlatTargetGenerator`  
For each K-means class, independently 40% chance to use flat constant (like v26_7) vs signed-alpha affine (like v26_6).  
**Manifold results (1151 synth samples — 40 subjects failed during generation):** 29/42 IND, recall 0.975, Vendi **3.553**, OOD norm 14.15. Confirms the mechanism: flat constants drive Vendi up by creating degenerate histogram spikes, but at the cost of 13 IND groups. Vendi increase here reflects weird piecewise-constant images, not useful training diversity.

## v26_15 — Double remap ⭐ worth keeping
**Generator:** `V26_15DoubleRemapTargetGenerator`  
Apply v26_6 twice in sequence: first remap gives y1 (normalized to [0,1]), second remap treats y1 as a new T1w input and remaps again. Compound non-linear transformation.  
**Manifold results:** **42/42 IND** (matches v26_6), recall 0.986, density 0.672, Vendi 2.765, OOD norm **17.95** (highest of all our methods, approaching ss_modeB at 18.30).  
**Why keep it:** Same perfect coverage as v26_6, but OOD samples are pushed further from all real clusters — the extra "stretch" of the OOD cloud may improve training robustness on edge cases. UMAP shows good manifold topology. Zero coverage cost.  
*Next: extract HOG features for v26_6 and v26_15 to evaluate texture-space diversity.*

---

## SynthSeg comparison variants

Implemented in `scripts/generate_synthseg_comparison.py`. Use SynthSeg's BrainGenerator with spatial augmentations disabled (so outputs remain in T1w space for atlas-based feature extraction).

### synthseg_modeA — Dense segmentation → BrainGenerator
SynthSeg v1.0 predicts a 32-label FreeSurfer segmentation from each T1w; BrainGenerator samples uniform GMM intensities U(25,225) per label.  
**Manifold results:** 13/42 IND, recall 0.742, Vendi **3.850**. Fixed anatomy (same spatial label map per subject) constrains spatial texture patterns → poor coverage. High Vendi reflects random GMM intensities on fixed anatomy creating diverse but unrealistic images.

### synthseg_modeB_em — EM cluster labels → BrainGenerator
sklearn GMM (K ∈ [3,10]) clusters T1w foreground → integer label map → BrainGenerator with uniform GMM intensities.  
**Manifold results:** 38/42 IND, recall **0.993**, Vendi 2.757, OOD norm **18.30**. Best recall, best exploration. Random spatial boundaries + fully unconstrained intensity assignment enables genuine contrast inversions. Our v26_6 beats it on coverage (42 vs 38 groups) and density.

---

## Key findings summary

| Method | IND groups | Recall | Vendi | OOD norm | Notes |
|---|---|---|---|---|---|
| v26_4 | 41 | 0.985 | 2.775 | 16.91 | baseline |
| **v26_6** | **42** | **0.989** | 2.802 | 17.16 | **best coverage, recommended** |
| v26_7 | 27 | 0.977 | 3.686 | 14.33 | high Vendi but degenerate |
| v26_9 | 41 | 0.990 | 2.816 | 17.40 | best Vendi in affine family |
| **v26_15** | **42** | 0.986 | 2.765 | **17.95** | **same coverage, more OOD exploration** |
| ss_modeB | 38 | 0.993 | 2.757 | 18.30 | best recall, worst coverage |

**Vendi note:** Vendi scores above ~3.0 in this feature space are driven by degenerate (flat/constant) regional histograms, not useful training diversity. IND coverage and recall are better proxies for training utility.

**Next frontier — HOG space:** All methods achieve <0.55 recall in HOG-972/HOG3D-512 feature spaces. HOG captures spatial gradient/texture patterns (scanner PSF, noise structure, tissue boundary sharpness) that intensity remapping fundamentally cannot change. Improving HOG coverage requires scanner simulation: Rician noise, k-space undersampling artifacts, B1 field inhomogeneity.
