# Texture / Structure-Preservation Analysis — Literature Review & Metric Choice

Level-1 (input-space, network-free) texture analysis for PALETTE. Goal: quantitatively
prove that PALETTE **preserves the source image's texture/gradient structure** while
SynthSeg (label-generative) **destroys it**. This file records how the field measures
"structure/texture preservation" and the tight metric set we adopt.

---

## How the field measures it

**1. Fidelity metrics — SSIM, PSNR, RMSE, intensity-PCC.**
Standard for image-to-image translation *when a ground-truth target exists* — they measure
how close the output is to a reference. **Not usable here:** we *intend* the contrast to
change, so SSIM/PSNR/RMSE would penalize exactly the augmentation we want. They only make
sense with a paired target image, which we don't have.

**2. Structure-preservation framing (harmonization / style-transfer).**
Harmonization explicitly aims to *preserve anatomical content while changing style/contrast*
— the same problem shape as ours. The measures used there are the relevant ones:
- **Mutual Information (MI)** between source and output — maximized in CycleGAN-based
  harmonization to preserve content; **contrast/modality-agnostic** (invariant to any
  monotone *or* inverted intensity mapping). This is the canonical content-preservation
  measure.
- SSIM / PCC — used but contrast-dependent, so weaker for our intentional-contrast-change case.

**3. Texture-specific perceptual metrics.**
- **FSIM** — feature similarity built on *phase congruency + gradient magnitude* (edges,
  textures, gradients). Motivates a gradient/edge-preservation measure.
- **DISTS** — weights texture similarity, tolerant to fine misalignment.
- **STSIM** — texture periodicity/directionality/granularity.
- **CW-SSIM** — invariant to small shifts/scale/rotation (complex-wavelet domain).
- **SAMScore** — deep semantic content-structural similarity (SAM features).
These are mostly perceptual, often 2D, and several are contrast-dependent — heavier than we need.

**4. Registration-derived local similarity.**
- **Local Normalized Cross-Correlation (LNCC)** — the standard similarity for
  multi-contrast/multi-modal registration; invariant to *local linear* intensity mapping →
  high iff local structure survives a contrast change. Directly matches "same texture,
  remapped contrast."

**5. Field-wide caveat that does NOT apply to us.**
Reference metrics are known to break under **spatial misalignment** between the two images.
In our setup the synthetic image is **voxel-aligned to its source by construction** (no
registration, spatial augs disabled), so this entire class of problems is avoided — a real
advantage worth stating in the paper.

---

## Our constraints → why most metrics don't fit

- **No ground-truth target** (augmentation, not translation-to-a-reference) → rules out SSIM/PSNR/RMSE.
- **Contrast is *intended* to change** → metric must be invariant to the intensity remap itself.
- **PALETTE uses signed-α (can invert)** → metric must be invariant to intensity *inversion*, not just scaling.
- **Voxel-aligned source↔synth** → we can use strong local/voxelwise measures (no misalignment penalty).
- **31 anatomical ROIs available** → measure preservation *per tissue class*, which also ties
  to the crush-case story (texture-defined regions).

---

## Adopted metric set (kept deliberately tight — every metric directly citable)

Audited each candidate for whether it is a *named, canonical* metric we can justify with a
one-line citation. Two clear the bar; the gradient-correlation I initially considered does
**not** (see "Rejected", below). Both adopted metrics are contrast-**and**-inversion-invariant
and computed **per 31-class ROI**, then pooled:

1. **(Normalized) Mutual Information — NMI** — `NMI(source, synth)` inside each anatomical class.
   The canonical "same anatomy, different intensity mapping" measure: MI for multimodal
   similarity = **Maes et al. 1997**; overlap-invariant **NMI = Studholme et al. 1999**.
   Invariant to any functional (monotone/inverted/nonlinear) intensity remap. High iff synth
   intensity is a deterministic function of source within the tissue (PALETTE); ≈ 0 for
   spatially-independent GMM noise (SynthSeg). *The content-preservation anchor.*
2. **|Local Normalized Cross-Correlation| — |LNCC|** — windowed source↔synth correlation
   within ROI. The primary similarity metric of ANTs/SyN registration (**Avants et al. 2008**);
   NCC is explicitly *invariant to linear brightness/contrast change*. We take `|·|` because
   PALETTE's signed-α can invert contrast locally. *The local-structure anchor.*

*(Optional 3rd, if we want a metric whose name is literally "texture":* **LBP histogram
similarity** — Local Binary Patterns, **Ojala et al. 2002**; rank/threshold-based, so more
intensity-invariant than GLCM — GLCM operates on raw intensities and is contrast-dependent,
so it is *not* suitable here. Computed slice-wise.)

**Rejected — gradient-magnitude correlation.** `corr(|∇source|, |∇synth|)` is intuitive but is
**not a named, citable metric** — it would be our own construction. The adjacent *named*
gradient metrics (FSIM, GMSD) are contrast-*sensitive*, so they penalize the intended contrast
change and don't fit. Dropped in favor of the two canonical measures above.

**Design choices we state transparently (standard-practice, not a cited protocol):**
- **Per-ROI computation** using the 31-class anatomical labels — to localize preservation to
  tissue classes (and tie to the crush-case story).
- **Eroded ROI masks** (drop 1–2 boundary voxels via `binary_erosion`). PALETTE's Voronoi
  sub-parcellation only *adds* false edges at region borders — it never *removes* real ones —
  so eroding borders removes that confound. Framed as **recall of source texture**, not
  symmetric similarity.

**Controls (defend against a "rigged metric" critique):**
- `auglab_default` and `gamma`/hist-eq — other *image-driven* augs; should also score high
  (they preserve texture too). This proves the metric rewards *any* texture-preservation, not
  something hand-tuned to PALETTE.
- `SynthSeg-EM` / `SynthSeg-noEM` — the texture-destroying references (expect MI ≈ 0).

---

## Framing for the paper — the 2×2 that makes PALETTE unique

Level-1 is **not** "PALETTE preserves texture, everyone else destroys it." Other image-driven
augs preserve texture too. The point is the **joint** picture:

| | Low contrast coverage | High contrast coverage |
|---|---|---|
| **Texture destroyed** | — | SynthSeg |
| **Texture preserved** | auglab_default / gamma | **PALETTE (alone)** |

- **Vertical axis (this analysis):** SynthSeg is the texture-destroying outlier (MI≈0);
  PALETTE + image-driven augs preserve.
- **Horizontal axis (contrast_manifold analysis):** plain intensity augs can't reach
  T2w-like inversions / full histogram coverage; PALETTE + SynthSeg can.
- **PALETTE is the only method in the winning quadrant** — and that explains *why* it beats
  both families downstream. This 2×2 is the intended headline figure.

---

## Sources

**Adopted-metric citations (primary):**
- **MI for multimodal registration** — Maes et al. 1997, *Multimodality image registration by
  maximization of mutual information*, IEEE TMI. (corroborated: https://pmc.ncbi.nlm.nih.gov/articles/PMC6560247/)
- **Normalized MI (overlap-invariant)** — Studholme et al. 1999, *An overlap invariant entropy
  measure of 3D medical image alignment*, Pattern Recognition.
- **LNCC / cross-correlation in registration** — Avants et al. 2008, *Symmetric diffeomorphic
  image registration with cross-correlation* (ANTs/SyN), Medical Image Analysis.
  https://www.sciencedirect.com/science/article/abs/pii/S1361841507000606
- **LBP (texture; more intensity-robust than GLCM)** — Ojala et al. 2002, IEEE TPAMI.
  https://www.sciencedirect.com/science/article/abs/pii/S1047320315001583

**Landscape / framing (secondary):**
- Similarity/quality metrics for MR image-to-image translation — https://www.nature.com/articles/s41598-025-87358-0 · https://arxiv.org/html/2405.08431v1
- HiFi-Syn (structure-preserving MR synthesis) — https://arxiv.org/pdf/2311.12461
- SAMScore (content structural similarity) — https://arxiv.org/html/2305.15367v2
- Deep learning for harmonization of structural MRI (survey; MI/SSIM/PCC for structure preservation) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11365220/
- Disentangled Latent Energy-Based Style Translation (MRI harmonization) — https://arxiv.org/html/2402.06875v1
