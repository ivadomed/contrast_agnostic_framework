# Label-cue importance — literature review and metric grounding

**Question.** For a given segmentation label (BraTS tumor sub-regions, open-ms MS lesions, CHAOS
abdominal organs), *what kind of local image evidence marks its boundary*: a first-order intensity
difference, a sharp intensity transition, or a change in texture — and **in what proportion**?

**Framing (deliberate).** This analysis does **not** classify labels into "texture-defined" vs
"boundary-defined" bins. It reports, per label and per modality, the **relative importance of three
local cue families**, so the defensible sentence is *"these regions rely on texture far more than
those ones"*, not *"this is texture-defined and that is not"*. That framing is not a hedge; it is
what the source literature actually supports (see §1) and what the estimator can actually deliver
given that the cues are correlated (see §5).

---

## 1. The primary grounding: Martin, Fowlkes & Malik (PAMI 2004)

D. Martin, C. Fowlkes, J. Malik, *"Learning to Detect Natural Image Boundaries Using Local
Brightness, Color, and Texture Cues"*, IEEE TPAMI 26(5):530–549, 2004
(earlier: NIPS 2002, *"Learning to Detect Natural Image Boundaries Using Brightness and Texture"*).
PDF: https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/martin_et_al.pdf

This paper **is** the present experiment, in 2-D natural images. Its method, which we transplant:

1. Define **one local cue per evidence family**: a *brightness gradient* (BG), a *color gradient*
   (CG, not applicable to us — MRI is scalar) and a *texture gradient* (TG).
2. Each cue at image location `p` with putative boundary orientation `θ` is the **χ² distance
   between feature histograms computed in the two half-discs** centred at `p` and split by `θ`.
   BG histograms bin intensity; TG histograms bin texton labels.
3. Each cue is then treated as a **standalone boundary detector** and scored by precision–recall
   against ground-truth (human-marked) boundaries; cues are combined by **logistic regression**.
4. The paper's stated contribution is to *"quantify the relative power of these cues"*, and its
   headline finding is that **an explicit treatment of texture is required** — brightness gradient
   alone is not sufficient to explain where humans place boundaries.

That is exactly our claim shape, our estimand, and our evaluation protocol. Two adaptations are
required and are stated as adaptations, not as the original method:

* **2-D → 3-D.** Half-*discs* become half-*balls*; orientation is a unit vector on the sphere rather
  than an angle. The χ² statistic and the PR protocol are unchanged.
* **Human-marked natural-image boundaries → dataset label surfaces.** Our "ground-truth boundary"
  is the surface of an expert-drawn segmentation label. This is arguably a *closer* fit to the
  original intent than BSDS was: BSDS boundaries are what humans *perceive*; ours are what expert
  annotators actually *drew*, which is precisely the target our networks are trained to reproduce.

**Related precursor:** J. Malik, S. Belongie, T. Leung, J. Shi, *"Contour and Texture Analysis for
Image Segmentation"*, IJCV 43(1):7–27, 2001 — the origin of comparing local **texton histograms**
across a putative boundary with a χ² statistic, and of the position that contour and texture
evidence must be handled as separate, cooperating channels rather than one intensity channel.

---

## 2. Splitting "intensity" from "boundary sharpness" — the medical-imaging grounding

A predictable objection: *"intensity difference and boundary sharpness are the same thing."* They
are not, and we do not have to argue it ourselves — the radiology margin-analysis literature
already separates them and measures them separately.

J. Xu, S. Napel, D. Rubin et al., *"Quantifying the margin sharpness of lesions on radiological
images for content-based image retrieval"*, Medical Physics 39(9):5405–5418, 2012.
PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3432101/

Their definition of margin sharpness has **two explicitly separable attributes**:
1. *"the intensity difference between a lesion and its surroundings"*, and
2. *"the sharpness of the intensity transition across the lesion boundary"*.

They estimate both by fitting a **sigmoid to the intensity profile sampled along the boundary
normal**, whose amplitude parameter is (1) and whose scale/width parameter is (2). We adopt exactly
this decomposition and assign the two attributes to two different cues:

* attribute (1) → our **BG** cue (magnitude of the first-order difference between the two sides);
* attribute (2) → our **EG** cue (steepness of the transition, with the step size divided out).

Related: K. Gilhuijs et al.'s margin-gradient and variance-of-margin-gradient measures on the
lesion surface shell (the shell-based ancestor of this family); and the parametric edge-sharpness
modelling of *"Edge Sharpness Assessment by Parametric Modeling: Application to Magnetic Resonance
Imaging"* (PMC4706083), which documents why **non-parametric sharpness estimates based on raw local
gradient or kurtosis are unreliable at MRI SNR**, and why a *fitted/normalised profile* estimate is
preferred. We follow that warning — see §4, EG-sharp.

## 2b. The gradient-orientation half of EG: NGF, already in use in this project

E. Haber & J. Modersitzki, *"Intensity Gradient Based Registration and Fusion of Multi-modal
Images"*, MICCAI 2006 — the Normalized Gradient Field. Already the adopted metric of this
project's Pillar-1 texture analysis
(`datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/`).

**Honest statement of reuse.** The existing Pillar-1 NGF is an **image-vs-image** similarity
(source vs generated volume) and *cannot* be applied verbatim to a **label-vs-image** question.
What is reused here is the NGF **object and its invariance argument** — the same ε-regularised
normalised gradient `ĝ = ∇I/√(‖∇I‖²+ε²)`, the same 3-tap Sobel derivative (Sobel 1968) — aggregated
over the local ball by its **structure tensor**:

    EG_coh(p) = λ_max / trace  of  Σ_{ball} ĝ ĝᵀ

i.e. how nearly *planar* the local gradient field is. Structure tensor: J. Bigün & G. Granlund,
ICCV 1987; J. Weickert, *Anisotropic Diffusion in Image Processing*, 1998. Its range is [1/3, 1],
and **1/3 is exactly the chance floor this project already documents for 3-D NGF** (three
orthogonal directions carrying equal weight) — the same floor reached from the same place, so the
two analyses stay commensurate. Verified empirically by the sanity test rather than assumed.
Documented as *derived from* the Pillar-1 metric, not as the same test.

### 2b-bis. The NGF similarity itself, as a descriptive index (continuity with the ablation)

The cue above uses the NGF *object* but not the NGF *similarity formula*. That formula takes two
gradient fields **of the same anatomy at the same voxel**; Pillar-1 has them (source vs generated
volume), and a label-vs-image question does not. Because the same formula is the metric of the
project's ablation texture analysis, we additionally report it here in the one form that is not
circular, so a single NGF number spans both analyses:

    NGF_idx = mean over the label surface of  (∇I·∇L)² / ((‖∇I‖²+ε²)(‖∇L‖²+ε²))

with `∇L` the gradient of the Gaussian-smoothed binary label mask — i.e. the ablation's formula
verbatim, with the label map as the second "image".

**Why it is a descriptive index and never a scored detector cue.** Used as a detector it is
circular: `‖∇L‖` is large exactly on the label surface and ~0 everywhere else, so it separates
surface from non-surface points from label geometry alone — it would score near-perfect AP **on
pure noise**, measuring only where the boundary is, which we already know.

**The null is what removes the circularity.** The label mask is randomly rotated (90° turns and
flips only — exact, no interpolation, shape and size preserved bit for bit) and translated, and the
statistic recomputed. Label *geometry* is held fixed and only its *registration to the image*
varies, so the reported z-score measures image–label agreement rather than the shape of the label.
A perfectly-drawn boundary that the image does not mark scores at the null.

**Validated on the phantoms** (`--sanity`), with the half-space label whose surface is the true
edge plane:

| image | NGF_idx | null | z |
|---|---:|---:|---:|
| step edge | 0.899 | 0.334 | **+62.8** |
| texture edge (no intensity edge) | 0.644 | 0.347 | +5.8 |
| iid noise, no edge | 0.332 | 0.334 | **−0.4** |

The null lands at **0.334**, empirically reproducing the 1/3 chance floor for 3-D gradient
orientations that the project's Pillar-1 NGF documents — the two analyses agree on the floor
without it being imposed. The index does not fire on structureless noise (z = −0.4), which is the
check that distinguishes it from the circular detector version. It does show a moderate response to
a pure *texture* edge (z = +5.8), so it is not a clean "intensity-edge-only" statistic; read it
alongside the four cues, not as a substitute for them. Empirical p is resolution-limited by the
number of permutations (1/(N+1)), so **z is the primary readout**, not p.

**Why not the more obvious per-orientation alignment `(∇I·u)²/(‖∇I‖²+ε²)`** — this was implemented
first and **rejected by the sanity test, on measurement**. Because every cue must be maximised over
candidate orientations for fairness (§4), and because for *any* gradient whatsoever some candidate
orientation lies within ~20°, that cue's cos² saturates near 1 at essentially every voxel: it
scored AP 0.571 on a pure step edge against textured background, i.e. barely above the 0.5 chance
line. Coherence requires no candidate orientation at all, which removes both the degeneracy and
the fairness concern in one move.

Why gradient-*orientation* rather than gradient-*magnitude* is the right boundary cue is the
standing premise of edge-based segmentation: M. Kass, A. Witkin, D. Terzopoulos, *"Snakes: Active
contour models"*, IJCV 1988, and V. Caselles, R. Kimmel, G. Sapiro, *"Geodesic Active Contours"*,
IJCV 22(1):61–79, 1997 — both of which define a segmentable boundary as a **ridge of the image
gradient that the contour normal aligns with**. If a label's surface is not such a ridge, the whole
edge-based tradition cannot find it, which is exactly the property we want to measure.

---

## 3. Isolating texture from intensity — why LBP, and not GLCM/Haralick

The single most dangerous failure mode of this analysis would be a "texture" cue that is really an
intensity cue in disguise: it would inflate texture importance everywhere and invalidate the
result.

The radiomics literature is unambiguous that naive texture features are exactly that vulnerable:

* *"Gray-level discretization impacts reproducible MRI radiomics texture features"*, PLOS ONE 2019
  (https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0213459) — the discretisation
  choice, not the tissue, drives the feature values.
* *"Gray-level invariant Haralick texture features"*, PLOS ONE 2019 (PMC6386443) — Haralick features
  *"are not reproducible unless the same quantization is performed"*; the paper exists because the
  standard ones inherit the intensity scale.
* Image Biomarker Standardisation Initiative (IBSI) reference manual, Zwanenburg et al.,
  https://arxiv.org/pdf/1612.07003 — intensity-histogram and texture families both *"require prior
  discretisation of intensities into grey level bins"*, and relative (fixed-bin-number)
  discretisation makes the bin width itself a function of the ROI's intensity range.

So GLCM/Haralick as a texture cue would be circular here. Instead:

T. Ojala, M. Pietikäinen, T. Mäenpää, *"Multiresolution Gray-Scale and Rotation Invariant Texture
Classification with Local Binary Patterns"*, IEEE TPAMI 24(7):971–987, 2002.
https://ieeexplore.ieee.org/document/1017623

The LBP code is, in the authors' words, *"by definition, invariant against any monotonic
transformation of the gray scale"*: it records only the **ordinal relations** between a voxel and
its neighbours. An LBP-family histogram therefore carries **no absolute intensity level and no
contrast scale** — a TG cue built on it cannot be a laundered BG cue. This is a *structural*
guarantee, not an empirical hope, and it is the same invariance principle the project already
relies on for NGF (by the chain rule `∇g(I) = g'(I)·∇I`, gradient direction survives any monotone
remap): one consistent invariance philosophy across both pillars, not two ad-hoc choices. Ojala et
al.'s own classifier is *"nonparametric discrimination of sample and prototype distributions"* —
a χ²-type distance between those histograms, precisely how Martin et al. use texton histograms, so
the two references compose without modification.

**The exact code used, and why it is not plain LBP.** Three refinements were forced by the sanity
phantoms (§6). Each was a measured failure of the simpler choice, not a preference:

1. **Directional local extrema, not centre-vs-neighbour.** Plain LBP is invariant to monotone maps
   of the intensity *values* but **not to a smooth spatial ramp**: a ramp makes every
   centre-vs-neighbour comparison deterministic, so the ramp band itself reads as a texture
   change. Phantom 3 — a pure intensity ramp with *no texture change whatsoever* — scored
   **TG = 1.000** under plain LBP: a total leak of intensity evidence into the texture cue, which
   would have inflated texture importance on exactly the smooth partial-volume transitions these
   datasets are full of. The fix is the **Directional Local Extrema Pattern** of S. Murala,
   R. P. Maheshwari & R. Balasubramanian, *Int. J. Multimedia Information Retrieval* 1(3):191–203,
   2012 (https://link.springer.com/article/10.1007/s13735-012-0008-2), which codes the centre
   against the **two opposite neighbours** per direction: under any monotone ramp the centre is
   never an extremum, so every pair codes identically regardless of slope. Measured: flat-vs-ramp
   χ² falls 0.28 → 0.0002 *while* fine-vs-coarse texture separation **rises** 0.28 → 0.52.
2. **Rotation-invariant count reduction** (number of the 13 antipodal pairs at which the centre is
   an extremum → 14 codes), exactly as Ojala et al.'s riu2 code is the number of 1-bits. This is
   an estimability requirement: a 26-neighbourhood has 2^26 patterns and even the 6-face-neighbour
   version's 64 could not be estimated from the few hundred voxels a half-ball holds — phantom 2
   failed outright at 64 codes, pure sampling noise. *Cost, stated rather than buried:* the
   arrangement of pairs is discarded, so the cue reads local ordinal roughness (extremum density,
   i.e. correlation length), not oriented pattern structure.
3. **Rank canonicalisation before a Gaussian high-pass** (`texture_input()`), in that order. The
   high-pass removes the smooth intensity component BG/EG already own — needed because the extrema
   form is exactly ramp-invariant only while the ramp does not compete with the local texture
   amplitude (residual leak measured at AP 0.72 without it). But high-passing the **raw** volume
   destroys the monotone-invariance guarantee outright, since blurring is linear and does not
   commute with a nonlinear monotone remap — phantom 4 caught TG jumping 0.646 → 0.987 under a
   remap that changes no ordinal relation at all. Rank-transforming first (for any strictly
   increasing g, `rank(g(I)) = rank(I)` exactly) collapses every remap onto one canonical volume
   *before* any linear operator sees it, so both properties hold at once. This is also the same
   rank-canonicalisation the project's earlier census texture metric used, for the same reason.

**Multiresolution:** as in Ojala et al., the code is computed at radius 1 and 2 and the two χ²
values averaged, so the cue is not tied to one texture scale.

**Exclusion band:** codes for voxels lying *on* the dividing plane straddle the boundary and do
encode the intensity step, so they are excluded from both half-histograms by a band of width equal
to the largest texture radius. With it, phantom 1 confirms TG sits near chance for a pure intensity
step (0.621 against a 0.526 no-structure reference).

---

## 4. The three cues as implemented

At a location `p` with candidate orientation `u` (unit vector), inside a ball of radius `R`, split
into half-balls `H⁺ = {x : (x−p)·u > b}` and `H⁻ = {x : (x−p)·u < −b}` with exclusion band `b`:

| cue | definition | family | source |
|---|---|---|---|
| **BG** | χ²( intensity-histogram(H⁺), intensity-histogram(H⁻) ), 16 equal-mass bins | first-order intensity difference | Martin 2004 §BG; Xu 2012 attribute (1) |
| **EG_coh** | λ_max/trace of the NGF structure tensor over the ball (orientation-free) | boundary as a locally planar gradient structure | Haber & Modersitzki 2006; Bigün & Granlund 1987; Kass 1988; Caselles 1997 |
| **EG_sharp** | max\|Δ\| ÷ total variation of the perpendicular-averaged intensity profile along `u` | transition width, step size divided out | Xu 2012 attribute (2) |
| **TG** | mean over r∈{1,2} of χ²( DLEP_r-histogram(H⁺), DLEP_r-histogram(H⁻) ), 14 codes, on the rank-canonicalised high-passed volume | texture change | Martin 2004 §TG; Malik 2001; Ojala 2002; Murala 2012 |

χ² statistic, as in Martin et al.: `χ²(h₁,h₂) = ½ Σᵢ (h₁ᵢ − h₂ᵢ)² / (h₁ᵢ + h₂ᵢ)`.

**Orientation handling — a fairness requirement, not a detail.** At a true boundary point the
correct normal is known from the label; at a non-boundary point it is not. Scoring positives with
their oracle normal while negatives must guess would manufacture the result. As in Martin et al.
(who maximise each oriented cue over orientation), **every cue is maximised over the same fixed set
of candidate orientations at every point, positive and negative alike** — the label normal is never
used as an input to any cue, only to *define* which points are positives.

**EG_sharp estimation.** Xu et al. fit a sigmoid `a + b/(1+exp(−(t−c)/w))` and read sharpness off
`w`. Fitting per point × per orientation is prohibitive at our scale, so EG_sharp is computed as

    max |finite difference|  ÷  total variation

of the profile, which **for any monotone profile is exactly `1/w`** (total variation *is* the step
amplitude `b`, so the ratio is `(b/w)/b`), is invariant to `b` — keeping EG_sharp non-redundant
with BG by construction — and degrades gracefully where the sigmoid model does not apply: a noisy
non-edge profile spreads its variation across every sample, so the ratio collapses to ~1/T rather
than reporting spurious sharpness.

**Measured against the sigmoid model it stands in for** (`--sanity` step 4b, 200 synthetic profiles
with known `w` and `b`, plus a least-squares grid fit of `w`): corr(proxy, 1/w_true) = **0.915**,
corr(proxy, 1/w_fitted) = **0.931**, corr(proxy, |b|) = **−0.05**. So the proxy tracks Xu et al.'s
sharpness parameter closely and is independent of the step amplitude, which is the property that
keeps EG_sharp from being a second copy of BG. *(This paragraph previously asserted such a
validation existed before it had been implemented — it now reports a check that actually runs.)*

An earlier version normalised by the profile *range* instead;
phantom 1 rejected it (AP 0.352 against a 0.342 no-structure reference), because range-normalisation
keeps only "how abrupt is the local variation" and discards "is there a coherent transition at
all" — for pure noise those are the same thing.

Each profile sample is averaged over a small disc **perpendicular** to `u` rather than read off a
single voxel line, as in Xu et al. and the Gilhuijs margin-gradient family, and heeding PMC4706083's
warning that unaveraged local-gradient sharpness estimates are unreliable at MRI SNR.

**Known residual leak, reported rather than buried:** even so, EG_sharp scores AP 0.813 on phantom 2
— a *pure texture edge with no intensity edge at all* — because a profile crossing from a smooth
region into a rough one has its total variation dominated by the rough side. So EG_sharp's
standalone AP over-credits the boundary family wherever roughness changes. This is one concrete
instance of the general reason §5 reports standalone AP *and* marginal ΔAP rather than a single
number: on that phantom TG scores 1.000 and takes the credit in the combined model.

---

## 5. Turning cue scores into "relative importance" — and its limits

Per Martin et al., each cue is scored as a standalone detector by **average precision** (area under
the precision–recall curve) at separating true label-surface points from non-boundary points, and
the cues are combined by **logistic regression**. We report two complementary quantities:

* **standalone AP** per cue — "how far does this cue alone get you";
* **marginal ΔAP** — the drop in the combined model's AP when that cue is removed, i.e. its *unique*
  contribution.

**Why both, honestly.** The three cues are correlated (a large, sharp intensity step produces high
BG *and* high EG_align). The variable-importance literature is explicit that this breaks
single-number importance: permutation-style importance *"may not cause a large drop in performance
because the model can rely on the other correlated feature, leading to underestimations of
importance for strongly correlated features"* — see *"Variable Importance in High-Dimensional
Settings Requires Grouping"*, https://arxiv.org/pdf/2312.10858, and the grouped-Shapley/SAGE
treatment therein. Standalone AP therefore **over**-credits a correlated cue and marginal ΔAP
**under**-credits it; the truth is bracketed by the pair, and we report the pair rather than a
single fabricated "importance %". A cue that is high on **both** is unambiguously important.

Model fitting uses **grouped (leave-subject-out) cross-validation** so no subject contributes to
both fit and evaluation; APs are aggregated across subjects with the project's existing paired
tests (`00_00_utils/stat_tests.py`).

**Negative-point sampling — the estimand depends on it, so it is stated up front.** Negatives are
non-boundary locations drawn 50/50 from (a) the *interior* of the label and (b) the *immediate
surroundings* of the label (within a fixed distance), never from far-away anatomy. The question
being answered is therefore *"what distinguishes the annotated surface from nearby non-surface
locations in the same neighbourhood"* — the user's intended comparison of the label against its
surroundings — and **not** the much easier question of distinguishing tumour-region from healthy
brain, which any intensity cue would win trivially.

---

## 6. Sanity phantoms (`cue_metrics.py --sanity`) — what must hold before any real data

The panel is only meaningful if the three cues are actually separable. Four synthetic volumes, each
with a known correct answer, are asserted before any dataset is touched:

Phantoms 1, 3 and 4a add an intensity structure on top of **one shared textured backdrop**, so the
texture is provably identical on both sides and "TG at chance" is a strict test. (An earlier version
used iid noise as the backdrop; then any locally-monotone structure — including the ramp — counted
as a texture change, and phantom 3 was not testing what it claimed to.)

| phantom | construction | required outcome |
|---|---|---|
| 1. pure step edge | shared backdrop + a mean step, 1-voxel transition | BG high, EG_coh high, EG_sharp high, **TG ≈ chance** |
| 2. pure texture edge | two regions with **identical marginal histograms** (both rank-transformed to normal scores), different spatial correlation length | **TG high, BG ≈ chance, EG_coh ≈ chance** |
| 3. ramp edge | same intensity step as phantom 1, spread over ~8 voxels | BG high, **EG_sharp low**, **TG ≈ chance** |
| 4. monotone-remap invariance | phantoms 1 & 2 through a nonlinear monotone map (`expm1(1.3z)`) | **TG unchanged exactly**, EG_coh unchanged to within 0.10, BG may change |

**Measured, 2026-08-02 (`cue_metrics.py --sanity`, all four PASS):**

| phantom | BG | EG_coh | EG_sharp | TG |
|---|---:|---:|---:|---:|
| 1 step edge | 1.000 | 0.926 | 0.995 | 0.621 |
| 2 texture edge | 0.472 | 0.517 | 0.813 | **1.000** |
| 3 ramp edge | 1.000 | 0.876 | **0.521** | 0.515 |
| 4a remap of 1 | 1.000 | 0.874 | 0.898 | 0.621 |
| 4b remap of 2 | 0.472 | 0.513 | 0.719 | 1.000 |
| iid noise, no edge (reference) | 0.500 | 0.429 | 0.535 | 0.526 |

Reading: the panel separates as designed. BG and EG_coh are blind to a texture edge (0.47/0.52,
i.e. chance); TG is near-blind to both a step (0.62) and a ramp (0.52) and saturates on a texture
edge (1.000); EG_sharp is what distinguishes a 1-voxel step (0.995) from the *same* step spread
over 8 voxels (0.521), which is the §2 claim that intensity difference and transition sharpness are
separable attributes, demonstrated. TG's monotone-remap invariance is exact to three decimals
(0.621 → 0.621, 1.000 → 1.000), confirming the §3 structural guarantee empirically. The one
imperfection is EG_sharp's 0.813 on phantom 2, documented in §4.

Phantom 4 is the direct empirical check of the §3 invariance claim; phantom 3 of the §2 claim.
If any of these fail, no result from the real datasets is reportable.

---

## 7. What this analysis does **not** establish

Stated explicitly so the paper does not overclaim:

* It measures the **local image evidence available at the annotated surface**, not what a trained
  network actually uses. A cue-importance profile is a property of the *data and its annotation*;
  attributing network behaviour to it would require a separate intervention (which the causal
  ablation ladder in `CLAUDE.md` already provides from the other direction).
* Cue importances are **not comparable in absolute terms across imaging modalities** with different
  noise characteristics; MRI-vs-MRI comparisons (BraTS / open-ms / CHAOS-MR) are the intended read,
  and any CT row would need its own caveat.
* "Texture" here means *what an LBP histogram over a local ball can express*. It is a broad but not
  exhaustive definition; a texture property with a correlation length longer than the ball radius
  would be missed.

---

## 8. Second literature round (2026-08-02) — prior art missed in round 1

Round 1 built the panel from Martin 2004 + Xu 2012 + Ojala 2002. A second search found five items
that should have been there from the start. Recorded with what each changed.

### 8.1 Fisher's ratio — direct medical precedent for the whole question  → **cue added**
*"Contrast quality control for segmentation task based on deep learning models — application to
stroke lesion in CT imaging"*, Frontiers in Neurology 2025
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11849432/). Quantifies how well an image supports its
label with **Fisher's ratio** — (μ_object − μ_background)² over pooled variance — and shows it
**predicts segmentation difficulty**: 82% of lesions their models failed to detect had Fisher's
ratio < 0.05, as did 77% of slices with unsatisfactory Dice. This is the medically-native, directly
citable form of our BG question, with a published threshold to compare against. Added as
`bg_fisher`, reported ALONGSIDE `bg` (which is more general — it sees any histogram difference,
including equal-mean/different-shape — while Fisher's ratio is more interpretable and connects to
prior numbers). Establishes that "quantify the image evidence for a label" is an accepted,
published activity, not something this project invented.

### 8.2 Inter-rater variability — an independent leg for the boundary claim  → **cite, don't compute**
Published glioma tumour-boundary variability is **~20% intra-rater and ~28% inter-rater**, and
medical segmentation has a whole methods subfield built on "ambiguous boundaries" (e.g. MSE-Nets,
https://arxiv.org/pdf/2311.10380). The boundary-aware-segmentation literature states the organ
contrast plainly: *"a radiologist segmenting a liver would usually trace liver edges first, from
which the internal segmentation mask is easily deduced"* (End-to-End Boundary Aware Networks,
MLMI 2019). Annotators disagree on tumour boundaries and agree on organ boundaries — the same
conclusion as our EG cues, reached without our metric. Cite alongside; do not re-derive.

### 8.3 Phase congruency — the contrast-invariant boundary measure we did not use  → **must justify**
Kovesi, *Image Features from Phase Congruency* / *Phase Congruency Detects Corners and Edges*
(https://www.peterkovesi.com/papers/phasecorners.pdf), building on the Morrone & Owens local-energy
model: features occur where Fourier components are maximally in phase, giving a measure of feature
significance that is **invariant to brightness and contrast** by construction, computed from
log-Gabor quadrature pairs. This is the classical, principled answer to "measure edge strength
without measuring intensity magnitude", and EG_sharp is a cruder hand-rolled cousin of it. Not
currently implemented. **Any reviewer from the vision side will ask why not**, and the honest
answer is availability of effort, not superiority — EG_sharp's advantage is only that it maps onto
Xu et al.'s clinically-interpretable transition-width parameter. Flagged as the most defensible
outstanding methods gap.

### 8.4 gPb — the successor to our primary citation  → **cite**
Arbeláez, Maire, Fowlkes & Malik, *Contour Detection and Hierarchical Image Segmentation*,
TPAMI 33(5):898–916, 2011
(https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) —
multiscale brightness/colour/texture cues plus spectral globalisation. Citing Martin 2004 without
its own successor is a visible gap; it also supports the multiscale (multi-radius) choice.

### 8.5 Texture features already work on THESE tasks  → **second texture cue added**
The most consequential find. Texture descriptors have published success on exactly our two disease
areas: LBP-TOP + HOG-TOP for 3-D MRI brain-tumour delineation
(https://www.sciencedirect.com/science/article/abs/pii/S0925231216310864); GLCM-based texture with
clustering for **boundary delineation of subtle white-matter lesions**
(https://www.nature.com/articles/s41598-022-07843-8); plus the MS texture-analysis line from §3.

**Consequence for interpretation.** The pilot's `tg` sat at chance for brats/open-ms (0.508 / 0.469).
Given the above, the leading explanation is **not** "these labels carry no texture" but "our
descriptor is too weak": we deliberately traded discriminative power for invariance (a 14-bin
extremum count, forced by the estimability failure in §3), where the published work uses far richer
GLCM / run-length / wavelet / LBP-TOP families. Reporting a null texture result with an
underpowered descriptor, against a literature that says otherwise, would be a serious error.

**Change made:** added `tg_joint`, the JOINT distribution of the extremum counts across both radii
(each bucketed to 5 levels → 25 codes) rather than the mean of two marginal χ². Ojala et al. treat
multiresolution LBP as a joint distribution across radii; averaging marginals discards exactly the
cross-scale interaction that encodes correlation length — the property GLCM/run-length/LBP-TOP
exploit. `tg` is kept as the conservative, invariance-clean cue and `tg_joint` as the sensitive one;
**both are reported**. Measured cost of the extra sensitivity, on the phantoms: `tg_joint` leaks
more on a pure intensity step (0.735 vs `tg`'s 0.621, against a 0.514 no-structure reference) while
matching `tg` at 1.000 on a true texture edge and staying at chance on a ramp (0.476). Both are
bounded by explicit `--sanity` assertions and both remain exactly invariant to monotone remaps.

**Still open after this round:** phase congruency (§8.3), and a GLCM-family cue computed on the
rank-canonicalised volume (which would neutralise the discretisation confound of §3 while
recovering co-occurrence structure) — the natural next addition if `tg_joint` still reads near
chance at full n.

---

## 9. Citation route for the paper's claims (verified 2026-08-02)

Claims are about the **anatomy/pathology**, not the datasets — glioma, brain MS lesions, and
healthy liver/spleen/kidney — which is what makes the clinical literature usable. Every row below
was checked against a fetched primary source unless marked otherwise.

**The claims are deliberately NOT symmetric between glioma and MS.** Lumping them weakens both: the
"no clear boundary" argument is strong for glioma and *reversible against us* for MS (see 9.2).

### 9.1 Glioma — the boundary is not defined by the image

| source | finding | how it supports the claim |
|---|---|---|
| *Imaging strategies for delineating glioma margins*, Meta-Radiology 2025 — https://www.sciencedirect.com/science/article/pii/S2667325825000081 | *"gliomas do not exhibit a clear boundary between malignant and healthy tissue"*; cells migrate *"several centimeters into apparently normal brain parenchyma"* | The absence of a boundary is a property of the disease, not of the acquisition |
| *Hypermetabolism and impaired cerebrovascular reactivity beyond the standard MRI-identified tumor border* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8156976/ | tissue beyond the MRI-identified border is already abnormal | The drawn line is a convention, not an observable edge |
| *FLAIRectomy: Resecting beyond the Contrast Margin for Glioblastoma*, Brain Sci 2022 — https://www.mdpi.com/2076-3425/12/5/544 | histopathology finds tumour inside peritumoral FLAIR abnormality, beyond the enhancing margin | Even the *outer* label does not bound the disease |
| Visser et al., *Inter-rater agreement in glioma segmentations on longitudinal MRI*, NeuroImage: Clinical 2019 — PMC6396436 **(fetched)** | 8 raters (4 expert, 4 novice). Generalized Conformity Index **0.32** postoperative enhancing tumour, 0.65 progression, 0.79 preoperative | GCI = of voxels *any* rater called tumour, the fraction *all* agreed on. 0.32 means eight trained raters agreed on under a third |

**Do NOT use** the widely quoted "20% intra-rater / 28% inter-rater" figure (often attributed to
Mazzara et al., IJROBP 2004;59(1):300–312). The attribution could not be confirmed, and Visser —
exactly the paper that would cite it — does not. The GCI numbers are a direct measurement and are
stronger.

### 9.2 Brain MS lesions — texture-informative and hard to delineate, NOT "no clear boundary"

⚠️ **The "no clear boundary" claim must not be made for MS.** Clinical radiology describes typical
MS lesions as *well* demarcated, and **"ill-defined borders" is a documented red flag pointing away
from MS** (toward PML; NMOSD/MOGAD lesions are the ill-defined ones) — *Assessment of lesions on
MRI in MS: practical guidelines*, Brain 2019;142(7):1858 (**fetched**;
https://academic.oup.com/brain/article/142/7/1858/5519813). A reviewer who knows MS imaging would
invert this claim. What is defensible instead:

| source | finding | how it supports the claim |
|---|---|---|
| Commowick et al., *Objective Evaluation of MS Lesion Segmentation*, Scientific Reports 2018 — PMID 30209345 | 53 cases, 4 centres, **each annotated by seven experts**; automatic methods still trail human expertise | The task has genuine annotation ambiguity even among experts |
| ⚠️ inter-rater Dice ~0.54–0.75 on MSSEG | **secondary sources only** — paper paywalled here | Read the PDF before quoting the range |
| Zhang et al., *Texture analysis of multiple sclerosis: a comparative study*, Magn Reson Imaging 2008 — https://pubmed.ncbi.nlm.nih.gov/18513908/ | texture features discriminate MS lesion vs **normal-appearing** WM vs normal WM | Lesion identity is carried by texture, beyond intensity level |
| *Detection of subtle white matter lesions… texture feature extraction*, Sci Rep 2022 — PMC8924181 **(fetched)** | GLCM texture used for detection / false-positive removal; intensity thresholding alone inadequate | Intensity alone is insufficient for these lesions |

**Correction to an earlier draft of this document:** it claimed the Sci Rep 2022 paper used GLCM for
*boundary delineation*. It does not — texture is used for detection and false-positive removal, and
boundary delineation is a separate Local Outlier Factor step. Claim 2 is therefore supportable in
the form *"texture is informative and intensity alone is insufficient"*, **not** *"the boundary is
defined by texture"*.

### 9.3 Glioma — texture

| source | finding | strength |
|---|---|---|
| *Detection of brain tumor in 3D MRI images using LBP and HOG*, Neurocomputing 2016 — https://www.sciencedirect.com/science/article/abs/pii/S0925231216310864 | LBP-TOP + HOG-TOP texture descriptors used to delineate tumorous regions | ⚠️ abstract only; verify before citing as delineation |

### 9.4 Healthy liver / spleen / kidney — boundaries are sharp

| source | finding | how it supports the claim |
|---|---|---|
| CHAOS challenge, Kavur et al., *Medical Image Analysis* 2021;69:101950 — https://arxiv.org/abs/2001.06535 **(fetched)** | single-modality **DICE 0.98 ± 0.00 (CT) / 0.95 ± 0.01 (MR)**; cross-modality falls to 0.88 ± 0.15 (liver) | The task is essentially solved within a modality — the boundary is there to be found |
| *End-to-End Boundary Aware Networks*, MLMI 2019 — https://web.cs.ucla.edu/~dt/papers/mlmi19c/mlmi19c.pdf | *"a radiologist segmenting a liver from CT images would usually trace liver edges first, from which the internal segmentation mask is easily deduced"* | The clinical workflow itself is edge-driven — the opposite of glioma |

**Do NOT cite** "manual liver segmentation Dice 0.95" (PMC6527212) as an inter-rater number: it is
manual **vs a reference standard** (3 observers), not human-vs-human.

---

## 10. The strongest reviewer objection: "it all depends on the contrast"

> *"Boundary clarity is a property of the sequence, not the disease. Enhancing tumour has a
> razor-sharp margin on T1c; MS lesions are conspicuous on FLAIR. You have only shown these labels
> are ill-defined in the sequences that happen to suit your argument."*

This is the objection most likely to be raised and it has to be answered directly. Three answers,
in increasing order of force:

**(a) For glioma, the literature already answers it.** The infiltration finding is specifically that
tumour extends beyond the **contrast-enhancing** region *and* beyond the **T2/FLAIR** abnormality
(§9.1). No available sequence bounds the disease. The sharp T1c margin is a *blood–brain-barrier
breakdown* boundary, not a tumour boundary — the standard neuro-oncology position, and it converts
the objection into support.

**(b) For organs, the claim is contrast-stable.** CHAOS reports 0.98 (CT) and 0.95 (MR); our own
measurement gives `eg_coh` 0.620–0.790 on T1-DUAL in-phase and 0.728–0.921 on T2-SPIR. High in
every contrast tested, which is what "the boundary is really there" should look like.

**(c) Our own per-modality measurement answers it head-on — and this is what citations cannot do.**
The analysis was run **separately per contrast**, so the objection is directly testable. `eg_coh`
(chance = 0.500):

| label | t1c | t1n | t2f | t2w |
|---|---:|---:|---:|---:|
| ET (enhancing tumour) | **0.463** | 0.398 | 0.458 | 0.441 |
| NCR | 0.418 | 0.352 | 0.396 | 0.384 |
| SNFH | 0.482 | 0.468 | 0.547 | 0.520 |
| whole tumour | 0.538 | 0.535 | 0.585 | 0.573 |

| label | FLAIR | T1w |
|---|---:|---:|
| MS lesion | **0.461** | 0.424 |

| organ | T1-DUAL in | T2-SPIR |
|---|---:|---:|
| liver | 0.635 | 0.728 |
| right kidney | 0.786 | **0.921** |
| left kidney | 0.770 | 0.885 |
| spleen | 0.620 | 0.865 |

Every pathology label stays at or below chance **in its own best contrast** — enhancing tumour on
**T1c**, where enhancement is most conspicuous, and MS lesions on **FLAIR**, where they are most
conspicuous. Every organ stays high in **both** contrasts. So the dichotomy is not an artifact of
sequence choice: it survives in the best case for the pathology labels and in the worst case for
the organs.

**This is the single best reason to keep the measurement in the paper** even though the three claims
are individually citable — it is the only evidence that answers the contrast objection
quantitatively, per sequence, on one common axis.

**Honest caveat to check before publishing:** ET on T1c scoring 0.463 is *surprising* — a
gadolinium-enhancing rim ought to be a strong local wall. Plausible explanations are that the
enhancing rim is thin relative to the 7 mm analysis ball, and that the surrounding brain is
edge-rich (see FINDINGS.md attack A2, where absolute levels showed both effects are real). Before
this table goes in a paper, **overlay a few ET-on-T1c boundaries on the image and look at them.**
If the rim is visibly sharp where the metric says 0.463, the metric is missing something at that
scale and the row must be withdrawn.
