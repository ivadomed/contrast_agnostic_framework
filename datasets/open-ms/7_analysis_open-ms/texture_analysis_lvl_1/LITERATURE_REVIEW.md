# Texture-Preservation Analysis (Level 1) — Metrics, Literature, and Justification

Level-1 = input-space, network-free. Goal: show that **image-driven augmentations preserve the
source image's texture, while label-generative synthesis (SynthSeg) destroys it** — the
mechanistic reason PALETTE beats SynthSeg on texture-defined structures downstream.

**The claim is categorical, not absolute:** SynthSeg sits at the no-texture floor; every
image-driven method (PALETTE *and* the conventional AugLab baseline) is far above it.

**2026-07-08 metric change:** the census/rank-correlation family (`census_r1`, `census_local8`)
is **retired** — its whole-ROI cross-region sign-cancellation (needing an arbitrary block-size
fix) made it unfit for the PALETTE-vs-auglab comparison, the question that matters most. The
**Normalized Gradient Field (NGF)** similarity — gradient-orientation agreement, established for
exactly this "compare structure across different contrast" problem in multimodal image
registration — is now the primary/only adopted metric. See §Adopted metric below; the retired
family is kept, unmodified, in §Superseded for the historical record and because `FINDINGS.md`'s
earlier conclusions cite it.

---

## The measurement problem

We compare a **source** T1w to an **intentionally contrast-randomized, possibly intensity-
inverted** version of it (PALETTE uses signed-α). A texture-*preservation* metric here must be
invariant to **monotone and inverted** intensity remaps — otherwise it scores the augmentation's
*intended* contrast change as "texture loss." That single requirement rules out most standard
tools (see next section).

**open-ms ROIs.** open-ms provides no anatomical parcellation — only a sparse binary MS-lesion
`dseg` (its brainmask is a computed extraction, not an annotation, so it is not used). We
therefore report **2 ROIs**: `lesion` (the FLAIR dseg, co-registered → applies to every contrast)
and `foreground` (intensity > 10% of the volume's p99 — a plain threshold stand-in for "brain",
**not** whole-image, since a whole-image comparison is confounded by background–background
correlation). This is the same annotation-faithful ROI choice as the open-ms coverage (Pillar-2)
analysis.

**A second measurement problem, found only after switching to a per-voxel metric: label-boundary
information is "given for free."** Every generation method is conditioned on the SAME real lesion
mask (it is the only real annotation open-ms provides), so the lesion/non-lesion **boundary
location** is handed to every method as ground truth — recovering it costs nothing and is not
preserved *texture*. A metric evaluated on the whole lesion mask therefore mixes two things: (a)
genuine interior structure that had to be reconstructed/preserved, and (b) trivial boundary
alignment that every method gets for free. See §Boundary-artifact diagnosis below for how this was
found and how far it can be corrected.

---

## Why not the standard texture / similarity metrics directly

- **Full-reference fidelity (SSIM, PSNR, RMSE):** assume the output should equal a reference →
  penalize the intended contrast change. Not applicable (no ground-truth target).
- **GLCM/Haralick, Gabor / wavelet energy, STSIM / DISTS / FSIM:** defined on intensities or
  their magnitudes → **contrast-sensitive** (a plain gamma would look like "texture change"). ✗
- **LBP (Ojala 2002):** invariant to monotone-*increasing* remaps, but **inversion complements
  the code** → not inversion-invariant → fails PALETTE's signed-α. ✗ (unless made sign-robust).
- **Gradient MAGNITUDE** (edge strength, GMSD-style): under a monotone remap g, the gradient scales
  by g′(x) — a spatially-varying, unknown factor — so raw gradient magnitude is exactly as
  contrast-sensitive as raw intensity. ✗ Gradient **orientation**, by contrast, is invariant to any
  *positive* per-voxel scaling (any locally-increasing g), and inversion (locally-decreasing g)
  only flips its sign — recoverable by squaring. This is the property multimodal image
  **registration** was built to exploit (two images of genuinely different contrast/modality that
  must still be compared structurally) — see §Adopted metric.

Two families survive contrast **and** inversion invariance: ordinal/rank-based (texture
literature; used previously here, see §Superseded) and gradient-**orientation**-based (multimodal
registration literature; adopted now).

---

## Adopted metric — Normalized Gradient Field (NGF) similarity

Per voxel, the **squared cosine of the angle between the source and generated gradient vectors**:

```
ngf(voxel) = (∇src · ∇gen)^2 / ((|∇src|^2 + ε^2)(|∇gen|^2 + ε^2))
```

**Citation and status.** This is the similarity measure of **Haber & Modersitzki (2006)**,
*"Intensity gradient based registration and fusion of multi-modal images,"* MICCAI — proposed
specifically for comparing images of **genuinely different contrast/modality** in multimodal image
registration. It is not a historical curiosity: the same formula is cited as still in active use in
the **2020 Learn2Reg challenge** (Häger et al., *"Variable Fraunhofer MEVIS RegLib Comprehensively
Applied to Learn2Reg Challenge,"* MICCAI workshop). The exact formula above was verified line-by-line
against a maintained open-source implementation
(github.com/BailiangJ/normalized_gradient_field_pytorch), not merely a paper description. A related
but **distinct** sibling in the same family, Gradient Correlation (Penney et al. 1998, "A Comparison
of Similarity Measures for Use in 2D-3D Medical Image Registration," MICCAI — described in
subsequent literature as seminal), computes normalized cross-correlation of gradient IMAGES; we use
Haber & Modersitzki's per-voxel squared-cosine form specifically, verified as above, not Penney's.

**Why gradient orientation rather than rank/census.** By the chain rule, ∇(g(x)) = g′(x)·∇x for any
differentiable g — so gradient DIRECTION at a voxel survives **any** locally-monotonic remap
(increasing *or* decreasing; decreasing flips the sign, absorbed by squaring), **regardless of how
nonlinear g is or how much it varies from region to region.** This invariance is pointwise: it needs
no block/window hyperparameter to avoid the cross-region sign-cancellation that made census_r1
under-score PALETTE and forced an ad hoc block-size fix (census_local8, now retired — see
§Superseded). NGF gets the same property for free from the gradient kernel's own small support.

**The eps (ε) parameter is fixed for numerical stability only — a small constant (1e-5 relative to
normalized [0,1] intensities), never tuned as an "edge sensitivity" knob.** (This project's earlier,
now-retired evaluation of NGF flagged the η/edge-parameter as "muddying interpretation" — that
concern was about treating ε as a *tunable* threshold. We instead report two explicit, untuned
aggregates — `ngf_all` and `ngf_edge`, defined next — rather than adjust ε to chase a threshold.)

- **`ngf_all`** = mean of `ngf(voxel)` over all ROI voxels.
- **`ngf_edge`** = mean restricted to the top 50% of ROI voxels by **source** gradient magnitude
  (`|∇src|²`) — a data-driven gate ("where the source actually has an edge"), not a tuned
  threshold, expressing "does the texture that exists survive" rather than being diluted by
  flat/uninformative voxels.

### The 1/3 chance floor — full derivation, verification, and honest citation status

For two **independent** 3-D gradient vectors, E[cos²(angle)] = 1/3 — **not 0**, unlike the
rank-correlation family's floor. This is not assumed; it is derived below, then verified two
independent ways (a 20-million-sample Monte Carlo, and the real `--sanity` phantom), and its
citation status is stated precisely rather than implied.

**Derivation (self-contained, three lines).** Let **v** be a unit vector uniformly random on the
sphere (an *isotropic* direction — no preferred orientation, exactly what i.i.d. per-voxel noise
produces), and **u** be *any other* vector, independent of v — fixed, or random with *any*
distribution, isotropic or not. Rotate coordinates so u points along the x-axis (allowed: v's
distribution is rotationally symmetric, so this doesn't change anything). Then
cos(θ) = u·v = v_x, so we need E[v_x²]. Since v is a unit vector, v_x²+v_y²+v_z² = 1 always; by the
sphere's symmetry E[v_x²] = E[v_y²] = E[v_z²]; so 3·E[v_x²] = 1, giving **E[v_x²] = 1/3**, hence
E[cos²θ] = 1/3. (Generalizes to E[cos²θ] = 1/d in d dimensions — this is why a 2-D toy version of
this derivation would give 1/2, not 1/3; ours is 3-D volumetric data.)

**Note the derivation only requires ONE side (v) to be isotropic — u can be anything, even fixed
or highly anisotropic.** This matters for real data: real MRI gradient directions are not
perfectly isotropic (anatomy has some preferred orientations), but that's fine — the noise-filled
side of the comparison (`synthseg_noem`, i.i.d. per-voxel Gaussian fill) *is* genuinely isotropic,
and that alone is enough to pin the expectation at 1/3 regardless of the source's own structure.

**Numerical verification (two independent checks, not one):**
1. `--sanity` phantom (a smoothed random 3-D volume, not real MRI): pure-noise case → 0.333.
2. A fresh 20,000,000-sample Monte Carlo of the abstract claim itself (independent of the
   codebase, plain NumPy): mean cos² = **0.333429** (SE 0.000067), matching 1/3 to 4 decimal
   places. Also tested the stronger claim needed for real data — u drawn from a *concentrated,
   non-uniform* distribution (not isotropic) against isotropic random v — still **0.333338**,
   confirming isotropy is only required on one side.
3. Real, independent data (30 real subjects, not simulated at all): `synthseg_noem`
   foreground-interior lands at **0.332 (FLAIR) / 0.319 (T1w)** — consistent with all of the above.

**Honest citation status.** The *math fact* — E[cos²θ]=1/3 for a coordinate of a uniformly random
point on the sphere S² (equivalently: the squared cosine of an isotropic 3-D direction follows a
Beta(1/2, 1) distribution) — is standard, textbook material in directional statistics (the study of
distributions of directions/orientations on spheres): see Fisher, Lewis & Embleton, *Statistical
Analysis of Spherical Data* (Cambridge, 1987), or Mardia & Jupp, *Directional Statistics* (Wiley,
2000), the field's standard references (since we square the cosine, direction and its reverse are
equivalent, making this technically "axial" rather than "vector" data — the same texts cover that
case). **What is NOT a citation we hold:** we have not found a paper stating "the chance floor for
NGF-style gradient-orientation image comparison is 1/3 in 3-D" as a specific, named fact for this
application — that application (applying a well-known geometric identity to characterize a texture
metric's floor) is ours, not lifted from a citable precedent. The math is not in question; its
specific use here has not been published elsewhere as far as we have found.

**Honest grounding status (same standard applied to census before it — no exemption):**
- **Solidly grounded, verified, current practice:** the core formula (Haber & Modersitzki 2006,
  code-verified); Sobel gradients (textbook); the squared-cosine framing for inversion invariance
  (standard description in the multimodal-registration literature).
- **Not from a paper — our own construction, stated plainly:** (1) NGF was designed as a
  registration **loss** (minimized while optimizing an alignment transform), not a post-hoc
  texture-**evaluation** metric — repurposing it this way is principled but not something we found
  a paper doing for exactly this augmentation-fidelity purpose. (2) The 1/3 floor is *our* symmetry
  derivation, confirmed empirically, not a cited fact. (3) The `ngf_edge` data-driven gate and the
  interior/shell boundary-artifact correction (next section) are our own diagnostic constructions,
  motivated by first-principles reasoning about this specific dataset and confirmed empirically —
  not lifted from a citable precedent. We do **not** claim "100% literature-grounded end-to-end."

**A second, mechanistically-independent corroborating metric was sought and not found — logged
here as due diligence, not swept under the rug.** Two candidates were prototyped and validated on
the same synthetic sanity protocol before any real data was touched, and both were **rejected on
clear mechanistic grounds**, not abandoned for lack of trying:
- **MIND** (Heinrich et al. 2012, *Modality independent neighbourhood descriptor for multi-modal
  deformable registration*, Medical Image Analysis 16(7)) — local self-similarity-patch descriptor,
  a genuinely different mechanism from gradient orientation. **Rejected:** its descriptor is
  variance-normalized, so two *independent* noise fields score 0.995 (should be low) — it cannot
  distinguish "the source's own noise realization" from "any other noise," exactly the case that
  matters most here (the SynthSeg/kmeans_voronoi floor).
- **Laplacian zero-crossing sign-agreement** (Marr & Hildreth 1980) — second-derivative-based,
  mechanistically distinct from NGF's first-derivative approach. **Rejected:** fails inversion
  invariance outright (a pure intensity inversion scores 0.000, the worst possible value, since
  Laplacian(1−x) = −Laplacian(x) exactly). Unlike gradient *vectors*, a Laplacian is a scalar, and
  the squared-cosine trick that gives NGF its invariance has no scalar analogue — fixing this would
  reintroduce the same block-size problem NGF was adopted specifically to avoid.

A candidate expected to work — local phase congruency (frequency-domain, provably contrast-
invariant by construction) — was identified but not built: a proper 3D multi-scale/multi-
orientation implementation is a substantial undertaking with real engineering risk in the time
available. **Decision (with the user): proceed with NGF as the sole adopted metric**, documented
plainly as single-metric, with this due-diligence search logged rather than omitted.

---

## Boundary-artifact diagnosis — the lesion-ROI limitation

Scoring the raw generated volumes surfaced a puzzle: `synthseg_noem` and `baseline_kmeans_label_remap_voronoi`
(both designed to sit at the noise floor) scored **well above 0.333** on the whole `lesion` ROI —
0.65 and 0.50 respectively — while `synthseg_noem` sat almost exactly **at** the floor (0.32–0.33)
on `foreground`. An interior (eroded ×3)-vs-boundary-shell decomposition (mirroring the erosion
diagnostic already used for the SynthSeg-blur finding) resolved it:

| ROI | method | interior | shell |
|---|---|---:|---:|
| foreground | synthseg_noem | 0.338 | 0.325 |
| lesion | synthseg_noem | 0.340 | 0.592 |
| lesion | synthseg_em | 0.320 | 0.642 |

**Mechanism:** the lesion boundary is the **one real label** open-ms provides, and every method is
conditioned on it — so recovering that boundary's *location* is free information, not preserved
texture, for every method equally. The `foreground` mask's *outer* (skull/background) edge is, by
contrast, genuinely **not** given to `synthseg_noem`: open-ms's coarse 2-label seg (lesion vs
everything else) never distinguishes background from brain, so noEM's uninformed fill has no way to
place that edge correctly — consistent with its foreground shell sitting at the floor rather than
above it. This is a strong, mutually-consistent confirmation of the mechanism, not a post-hoc story
fit to one number.

**What this means for reporting, and its limit.** `foreground` erosion (×3) is fully powered — all
30 subjects retain ≥200 interior voxels (open-ms's foreground spans the majority of the image, so
a thin erosion shell is a negligible fraction). **`lesion` erosion is not viable**: MS lesions are
small, thin, and often disconnected; erode(×3) leaves only 6/30 subjects with any interior voxels at
all, and the ≥200-voxel qualifying threshold used in the diagnostic script passed **only ~2
subjects** — a severely underpowered, size-biased sample (only unusually large lesions survive),
not a trustworthy comparison. **We therefore report `foreground`-interior NGF as the properly-
powered, boundary-artifact-excluded texture measure, and report whole-`lesion` NGF only with this
explicit caveat** (it partially reflects free boundary information, more so for methods with no
real-intensity-informed partition). An earlier draft of this analysis briefly reported a
palette-vs-auglab lesion comparison from the erode(×3) diagnostic (n≈2); that comparison is
**retracted** as unpowered, not as wrong — it may be right, but it cannot be claimed from that data.

---

## Results

**Scope decision: `foreground` only — general structure preservation, not MS-lesion-specific
texture.** The project's claim is that palette preserves source structure in general; MS-lesion
texture specifically is not the target and is not reported (see the box below for why `lesion`
would be a poor fit for that claim even if it were). Plots (one file per contrast, no reference
line drawn — see note below): `outputs/plots/ngf_headline_{flair,t1w}.png` and
`outputs/plots/ngf_violin_foreground_interior_{flair,t1w}.png` (per-subject distributions —
visually confirms the paired-Wilcoxon result below, especially T1w where palette's whole
distribution sits above auglab's). Generated by `scripts/plot_ngf_texture.py`. Retired census
plots/tables are in `outputs/archive_census/` (historical only).

> **Why `foreground` is a whole-brain measure, not a lesion-excluding one.** `foreground` is a
> plain intensity threshold with no anatomical knowledge — it includes lesion tissue as an
> undifferentiated part of the mask (verified: **100% of lesion voxels fall inside `foreground`**,
> expected since MS lesions are, by the sequence's design, hyperintense on FLAIR — Hajnal et al.
> 1992 introduce FLAIR; its use for MS lesion conspicuity is now clinical standard, cf. Filippi et
> al. 2016 MAGNIMS consensus; the source dataset itself is Lesjak et al. 2018, *A Novel Public MR
> Image Dataset of Multiple Sclerosis Patients With Lesion Segmentations*, Neuroinformatics).
> After erosion, lesion voxels are still present but **diluted to 1.6%** of all foreground-interior
> voxels pooled across the cohort — the reported number is a **whole-brain** structure-preservation
> score with lesion tissue folded in as a small, undifferentiated minority, not a targeted
> lesion measure. A genuinely lesion-targeted number would be the whole-`lesion` ROI, which is
> exactly the one with the unresolved boundary-artifact confound (§Boundary-artifact diagnosis) —
> consistent with this project not reporting it, since it isn't the claim being made anyway.

**No reference/floor line is drawn on the plots (by request)** — the bars/violins carry the
comparison on their own. The 1/3 chance-floor derivation (E[cos²θ] for independent 3-D unit
vectors) and its empirical confirmation on real data (`synthseg_noem` landing at 0.319–0.332, see
table below) remain documented here as part of the metric's validation, for anyone who wants the
reasoning behind why those absolute values are interpreted as "no texture" rather than "some
texture" — it just isn't drawn as a line on the chart.

Final, properly-powered NGF (`ngf_all`; 30 subjects × 10 variants = n=300 volumes/cell), noblur
set, `foreground`-interior (erode ×3; 30/30 subjects retained):

| method | FLAIR | T1w | reading |
|---|---:|---:|---|
| palette | 0.808 | 0.894 | structure preserved, highest |
| auglab_default | 0.772 | 0.777 | structure preserved, second |
| synthseg_em | 0.461 | 0.422 | above floor (real-informed partition, random fill) |
| baseline_kmeans_label_remap_voronoi | 0.453 | 0.401 | above floor (real-informed partition, random fill) |
| synthseg_noem | 0.332 | 0.319 | matches the derived 1/3 chance floor |

**Paired Wilcoxon, palette vs auglab_default, per-subject (n=30):** FLAIR: median 0.813 vs 0.779,
rank-biserial effect **+0.548**, 21/30 subjects favor palette, **p = 7.6e-3**. T1w: median 0.903 vs
0.777, effect **+0.901**, 29/30 subjects favor palette, **p = 1.2e-6**. **Palette significantly
preserves more structure than conventional augmentation**, decisively in T1w and reliably in
FLAIR — the reverse of the retired census family's conclusion (§Superseded), now on a metric with
no cross-region cancellation artifact to explain away.

**Causal chain, confirmed and now more precise than the retired census version:** palette (real
fill + real-informed partition) ≫ synthseg_em / baseline_kmeans_label_remap_voronoi (real-informed partition, **random**
fill, ~0.40–0.52) > synthseg_noem (uninformed partition, random fill, **0.32–0.33**, matching the
derived 1/3 chance floor). The fill carries most of the structure; a real-intensity-informed
partition alone (K-means or EM, no real fill) contributes a modest, consistent residual; an
uninformed partition contributes nothing above chance.

*(The whole-`lesion` NGF numbers computed during the investigation — where palette and auglab come
out ~tied — are kept in `FINDINGS.md`'s reasoning trail, not here, precisely because that ROI
carries the boundary-given-for-free confound and isn't the claim this project makes; do not read
those numbers as contradicting the table above.)*

### Confirmatory result — the usual (with-blur) deployed configuration

The noblur set isolates the fill from blur, but a natural question is whether the result survives
under the configuration actually used in training (blur on). Before running this, we found the
**with-blur configs still had the spatial-augmentation confound** (`p_rotation=p_scaling=0.2` for
synthseg_em/noem/auglab_default, 0 for palette/baseline_kmeans_label_remap_voronoi — same issue as the noblur set, just
never fixed there). Fixed the same way: new configs
(`data/configs_blur_nospatial/`) zero spatial for the 3 affected methods only, blur left untouched;
regenerated those 3 methods in place in `data/generated/` (palette/baseline_kmeans_label_remap_voronoi already correct,
untouched).

**`foreground`-interior NGF, with blur, spatial-matched, n=30×10:**

| method | FLAIR | T1w |
|---|---:|---:|
| palette | 0.764 | 0.859 |
| auglab_default | 0.703 | 0.727 |
| baseline_kmeans_label_remap_voronoi | 0.466 | 0.416 |
| synthseg_em | 0.405 | 0.407 |
| synthseg_noem | 0.332 | 0.324 |

**Paired Wilcoxon, palette vs auglab_default (n=30):** FLAIR effect=+0.652, 23/30 subjects,
**p=1.2e-3**; T1w effect=+0.996, 29/30 subjects, **p=3.7e-9**. **The gap is larger and more
significant with blur than without** (noblur: +0.548/+0.901, p=7.6e-3/1.2e-6) — not an artifact of
the isolated no-blur ablation. `synthseg_noem` is essentially unchanged (0.332/0.324 vs 0.332/0.319
noblur), still at the derived floor — blur does not add real texture to an already texture-blind
fill, as expected. This is the strongest, most training-representative confirmation of the headline
result.

---

## Honest caveats (state these in the paper)

- **No input-space metric puts image-driven methods near ceiling (1.0).** Real augmentations
  genuinely restructure contrast (piecewise remaps, mild blur) — a moderate-but-far-from-floor
  score is the honest signature of "real, non-trivial augmentation that still derives from the
  source," not a metric flaw. The defensible claim is the categorical gap (image-driven ≫
  SynthSeg-noEM ≈ chance floor), robust across ROIs.
- **The `lesion` ROI cannot be cleanly separated into interior/boundary** (erosion destroys most
  lesions — see §Boundary-artifact diagnosis) — whole-`lesion` NGF partially reflects boundary
  information every method receives for free, more so for methods with no real-intensity-informed
  partition (synthseg_noem) than those with one (synthseg_em, palette-family). Read whole-`lesion`
  numbers with this in mind; `foreground`-interior is the clean comparison.
- **The phantom (sanity harness) is for metric validation only** (identity→ceiling, monotone
  remaps→ceiling, inversion→ceiling, noise→the derived 1/3 floor), *not* for predicting real
  volumes' absolute values — trust the real-data run for magnitudes.
- **Level-1 is correlational supporting evidence.** It shows image-driven methods preserve more
  source structure than SynthSeg-noEM — not that texture preservation *causes* the downstream
  segmentation win. The causal claim rests on the **baseline_kmeans_label_remap_voronoi ablation** (same K-means/Voronoi
  partition as PALETTE, fill replaced by noise, partition held fixed).

---

## Parameter / design choices — justification (none tuned per result)

- **Sobel gradient kernel, 3×3×3, reflect/replicate-padded.** Standard first-derivative kernel;
  same family used by every NGF reference implementation consulted.
- **ε = 1e-5 (numerical stability only).** Not tuned as an edge-sensitivity threshold — see
  §Adopted metric for why that would repeat this project's earlier, retired NGF concern.
- **`ngf_edge` = top 50% of ROI voxels by source gradient magnitude.** A data-driven median split,
  not a value chosen to maximize any gap.
- **Reported ROI = `foreground` only (not `lesion`).** `lesion` is open-ms's only real annotation
  and was computed during the investigation (§Boundary-artifact diagnosis, `FINDINGS.md`), but this
  project's claim is general structure preservation, not MS-lesion-specific texture, and `lesion`
  additionally carries the unresolved boundary-given-for-free confound — so it is not a reported
  result. (on-harmony's now-superseded census run used 31 anatomical labels over the whole brain,
  same spirit as `foreground` here.)
- **`foreground` erosion = 3 voxels (interior-only reporting).** Chosen entirely on voxel-count
  feasibility (§Boundary-artifact diagnosis: 30/30 subjects retain ≥200 interior voxels), not on
  which choice produces a more favorable result. `lesion` cannot survive the same erosion (kills
  most lesions) — a second, independent reason it isn't the reported ROI.
- **No floor/reference line drawn on plots (author preference).** The 1/3 chance-floor derivation
  and its empirical confirmation remain in §Adopted metric and §Results as text, not a plot
  element — a communication choice, not a change to the underlying math.

---

## Is this fully grounded / not hacked? — honest answer (no exemption from this standard)

**Not a 100% guarantee — here is precisely where the line falls, so no one has to take our word
for the rest.**

- **Solidly grounded, verified against source, current practice:** the NGF formula itself —
  identical to Haber & Modersitzki (2006), verified against a maintained reference implementation
  line-by-line, not just a paper description; still cited in active use as of the 2020 Learn2Reg
  challenge; Sobel gradients (textbook); squared-cosine as the standard way to get inversion
  invariance from a gradient-angle comparison (standard description in the multimodal-registration
  literature).
- **Not from a single paper — our own principled composition, stated plainly, not hidden:**
  (1) NGF is a registration **loss**; using it as a post-hoc texture-**evaluation** metric for
  augmentation fidelity is a repurposing we have not found published elsewhere for this exact
  purpose. (2) The 1/3 chance floor is *our derivation* (confirmed empirically, not cited).
  (3) The `ngf_edge` gate and the interior/shell boundary-artifact correction are our own
  diagnostic constructions for this dataset, motivated by first-principles reasoning and confirmed
  empirically (§Boundary-artifact diagnosis), not lifted from a citable precedent. We do **not**
  claim "100% literature-grounded end-to-end" — no post-hoc evaluation metric assembled for a
  specific dataset can honestly claim that.
- **Not hacked:** the metric was chosen on *principle* (pointwise contrast+inversion invariance,
  no block-size hyperparameter — directly motivated by the specific failure mode found in the
  now-retired census family) and validated on synthetic controls (identity→ceiling, monotone
  remaps→ceiling, inversion→ceiling, noise→the *derived* 1/3 floor) **before** any real generated
  volume was scored. The boundary-artifact finding was itself discovered by scrutinizing an
  unexpected real-data result rather than accepting it — see §Boundary-artifact diagnosis.

---

## Superseded — census/rank-correlation family (retired 2026-07-08)

**Kept verbatim for the historical record and because `FINDINGS.md`'s earlier conclusions
(Anomalies 1 and 2, the SynthSeg-blur and PALETTE-vs-auglab sign-cancellation findings) were
derived using it. Do not use for new conclusions — read §Adopted metric instead.**

### Rank/Census-transform correlation (texture) — retired
The **rank (census) transform** (Zabih & Woodfill 1994) replaces each voxel by the fraction of
its neighbours it exceeds — an ordinal, **LBP-family** local encoding. It is invariant to any
monotone-increasing intensity map *by construction*, and Zabih & Woodfill introduced it
**specifically to make correlation-based matching robust to intensity change** — so "rank
transform + correlation" is the metric's original intended use, not a repurposing.

- **Metric:** `|corr( rank(source), rank(synth) )|` within each eroded ROI. The `|·|` makes it
  **inversion-invariant** (inversion flips the rank field's sign; correlation → −1 → |·| → 1).
- **Floor = 0:** the correlation of two independent rank fields is 0, so texture-destroyed noise
  (SynthSeg) lands at ~0 — a clean, interpretable "no texture" baseline.
- **Window radius r** (neighbourhood): reported across **r ∈ {1, 2}** for robustness (not fixed
  to one value). r=1 is maximally local (most stringent); r=2 is less sensitive to sub-voxel blur.
- **Ties** (equal intensities, e.g. flat SynthSeg regions) are handled by strict `>` — flat
  regions correctly yield near-zero correlation (no texture to preserve).

### `census_local8` (cancellation-robust, windowed variant) — retired
Same rank field as census_r1 (radius r=1), but the `|corr|` is taken **within non-overlapping 8³
blocks and averaged** (voxel-weighted) rather than once over the whole ROI.

**Grounding — this is the SSIM structure term on rank-transformed images.** SSIM (Wang et al. 2004)
factors into luminance × contrast × *structure*, and its **structure term
`s(x,y) = (σ_xy + C)/(σ_x σ_y + C)` is exactly a local windowed correlation coefficient**, computed
in a local window (classically an 8×8 square — matching our block size — or an 11×11 Gaussian) and
averaged over the image ("mean SSIM"). census_local8 *is* that structure term, with two principled
substitutions: (i) inputs are **census/rank-transformed** (Zabih & Woodfill 1994) so the correlation
is contrast-invariant and ordinal rather than raw-intensity (a rank/Spearman analogue of SSIM's
Pearson structure term), and (ii) it is **absolute-valued** so PALETTE's signed-α inversions score
as preserved. The same windowed-local-correlation idea also appears as local normalized
cross-correlation (LNCC) in registration (Avants et al. 2008, ANTs/SyN).

*Why it was introduced.* Whole-ROI `|corr|` sums *signed* products across the entire ROI **before**
`|·|`. PALETTE's signed per-region α (`fromSeg.py`) makes some regions +correlated and others
−correlated with the source; over a large ROI these **cancel** in the sum, so census_r1
under-scores PALETTE for doing exactly what it is designed to do (randomize contrast direction
per region). Windowing confines each correlation to ≈one region, so the `|·|` is applied per block
and the signs no longer cancel.

**Why it was retired, not just supplemented.** The 8³ block size is an arbitrary hyperparameter
trading cancellation-robustness against a rising chance floor (~0.05–0.15 vs ~0.007 whole-ROI) —
and even with it, the census family could not cleanly resolve the PALETTE-vs-auglab comparison
(the question that matters most): census_r1 said auglab≥palette (the cancellation artifact),
census_local8 said they were ~tied. NGF's pointwise invariance (§Adopted metric) needs no such
block-size compromise at all — the underlying problem is solved rather than mitigated.

### Normalized Mutual Information — dropped before the full retirement
`NMI = (H(X)+H(Y))/H(X,Y) ∈ [1,2]` (Maes 1997; Studholme 1999) was the on-harmony "content"
secondary metric, already removed from the open-ms analysis before the census retirement above:
it is computed on the 1-D set of ROI intensity *values* (ignoring their spatial arrangement), so it
is **spatially blind — not a texture metric**. It reflects intensity-mapping determinism, not local
structure preservation, and was briefly (and wrongly) used to argue a PALETTE>auglab ordering; that
argument was withdrawn. NMI remains only in the shared on-harmony pipeline.

---

## Framing — the 2×2 that makes PALETTE unique

Level-1 alone is not the whole story (other image-driven augs also preserve texture). Combined
with the contrast-manifold coverage analysis:

| | Low contrast coverage | High contrast coverage |
|---|---|---|
| **Texture destroyed** | — | SynthSeg |
| **Texture preserved** | auglab_default / gamma | **PALETTE (alone)** |

- Vertical axis = this analysis (SynthSeg destroys texture; image-driven preserve).
- Horizontal axis = contrast-manifold (plain intensity augs under-cover; PALETTE + SynthSeg cover).
- **PALETTE is the only method in the winning quadrant** — the intended headline figure.

---

## Sources

**Primary (adopted metric — NGF):**
- Haber, E. & Modersitzki, J. (2006). *Intensity gradient based registration and fusion of
  multi-modal images.* MICCAI, pp. 726-733. — the NGF formula, verified against a maintained
  implementation: github.com/BailiangJ/normalized_gradient_field_pytorch.
- Häger, S. et al. (2020). *Variable Fraunhofer MEVIS RegLib Comprehensively Applied to Learn2Reg
  Challenge.* MICCAI workshop. — confirms NGF's continued current use.
- Penney, G. P. et al. (1998). *A Comparison of Similarity Measures for Use in 2D-3D Medical Image
  Registration.* MICCAI. — Gradient Correlation, a related-but-distinct sibling metric (see
  §Adopted metric for why we use Haber & Modersitzki's form specifically, not this one).
- Fisher, N. I., Lewis, T. & Embleton, B. J. J. (1987). *Statistical Analysis of Spherical Data.*
  Cambridge University Press. — standard reference for the 1/3 chance-floor derivation (§The 1/3
  chance floor); the math fact is standard, its use to characterize NGF's floor here is ours.
- Mardia, K. V. & Jupp, P. E. (2000). *Directional Statistics.* Wiley. — same role as above,
  covers the "axial" (sign-free) case relevant to a squared cosine.

**Considered, unsuitable here (contrast/inversion-sensitive):**
- SSIM/PSNR/RMSE (full-reference; assume a ground-truth target) — Wang et al. 2004, IEEE TIP.
- GLCM/Haralick — Haralick et al. 1973, IEEE TSMC.
- Gabor / wavelet texture energy; FSIM, CW-SSIM, DISTS, STSIM.
- LBP — Ojala et al. 2002, IEEE TPAMI (monotone-invariant but not inversion-invariant).

**Superseded (retired 2026-07-08 — kept for historical citation, see §Superseded):**
- Rank/census transform — Zabih & Woodfill 1994, *Non-parametric local transforms for computing
  visual correspondence*, ECCV.
- SSIM local-window aggregation (grounded census_local8) — Wang et al. 2004, IEEE TIP; local
  normalized cross-correlation — Avants et al. 2008 (ANTs/SyN), Med. Image Anal.
- MI / NMI (on-harmony only; already dropped for open-ms before the full census retirement) — Maes
  et al. 1997, IEEE TMI; Studholme et al. 1999, Pattern Recognition.

**Landscape / framing:**
- MR image-to-image translation metrics — https://www.nature.com/articles/s41598-025-87358-0
- DL harmonization of structural MRI (structure-preservation via MI/SSIM/PCC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11365220/
