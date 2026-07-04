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

## Adopted metric set (kept deliberately tight)

Two complementary, contrast-**and**-inversion-invariant metrics, computed **per 31-class ROI**
and pooled:

1. **Within-ROI Mutual Information** — `MI(source, synth)` inside each anatomical class.
   The literature-standard content-preservation measure. Invariant to any functional
   (monotone/inverted/nonlinear) intensity remap. High iff synth intensity is a deterministic
   function of source within the tissue (PALETTE); ≈ 0 for spatially-independent GMM noise
   (SynthSeg). *The rigorous one.*
2. **Gradient/edge preservation** — Pearson correlation of **gradient-magnitude** maps,
   `corr(|∇source|, |∇synth|)`, within ROI. FSIM-family; uses magnitude → inversion-invariant.
   *The intuitive "do the edges survive?" one.*

*(Optional 3rd if space allows: `|LNCC|` — local, linear-invariant structural similarity.)*

**Boundary-artifact handling:** compute per-ROI on **eroded** masks (drop 1–2 boundary voxels).
PALETTE's Voronoi sub-parcellation only *adds* false edges at region borders — it never
*removes* real ones — so eroding borders removes that confound. Framed as **recall of source
texture**, not symmetric similarity.

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
- Similarity/quality metrics for MR image-to-image translation — https://www.nature.com/articles/s41598-025-87358-0 · https://arxiv.org/html/2405.08431v1
- HiFi-Syn (structure-preserving MR synthesis) — https://arxiv.org/pdf/2311.12461
- SAMScore (content structural similarity) — https://arxiv.org/html/2305.15367v2
- Deep learning for harmonization of structural MRI (survey; MI/SSIM/PCC for structure preservation) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11365220/
- Disentangled Latent Energy-Based Style Translation (MRI harmonization) — https://arxiv.org/html/2402.06875v1
