# Texture-Preservation Analysis (Level 1) — Metrics, Literature, and Justification

Level-1 = input-space, network-free. Goal: show that **image-driven augmentations preserve the
source image's texture, while label-generative synthesis (SynthSeg) destroys it** — the
mechanistic reason PALETTE beats SynthSeg on texture-defined structures downstream.

**The claim is categorical, not absolute:** SynthSeg sits at the no-texture floor; every
image-driven method (PALETTE *and* the conventional AugLab baseline) is far above it.

---

## The measurement problem

We compare a **source** T1w to an **intentionally contrast-randomized, possibly intensity-
inverted** version of it (PALETTE uses signed-α). A texture-*preservation* metric here must be
invariant to **monotone and inverted** intensity remaps — otherwise it scores the augmentation's
*intended* contrast change as "texture loss." That single requirement rules out most standard
tools (see next section). Metrics are computed **per anatomical ROI** (31 classes), on **eroded**
masks (drop the tissue borders shared by all methods).

---

## Why not the standard texture / similarity metrics directly

- **Full-reference fidelity (SSIM, PSNR, RMSE):** assume the output should equal a reference →
  penalize the intended contrast change. Not applicable (no ground-truth target).
- **GLCM/Haralick, Gabor / wavelet energy, STSIM / DISTS / FSIM:** defined on intensities or
  their magnitudes → **contrast-sensitive** (a plain gamma would look like "texture change"). ✗
- **LBP (Ojala 2002):** invariant to monotone-*increasing* remaps, but **inversion complements
  the code** → not inversion-invariant → fails PALETTE's signed-α. ✗ (unless made sign-robust —
  which is exactly what we do below).

The family that survives contrast **and** inversion invariance is ordinal/structural. We use the
ordinal one, from the texture literature.

---

## Adopted metrics

### Primary — Rank/Census-transform correlation (texture)
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

### Secondary — Normalized Mutual Information (content)
`NMI(source, synth) = (H(X)+H(Y))/H(X,Y) ∈ [1,2]` per ROI (Maes 1997; Studholme 1999). Invariant
to *any* intensity remap; a whole-region statistical-dependence check complementing the local,
ordinal census metric. Note it is diluted for PALETTE by design (per-Voronoi-cell remaps make the
per-ROI source→synth map piecewise), so it too is read relatively.

*(NGF and whole-ROI LNCC were evaluated and dropped: NGF added an η edge-parameter that muddied
interpretation, and LNCC conflates coarse contrast layout with fine texture. Two clean metrics
are clearer for reviewers than four.)*

---

## Results (real data — preliminary 2-subject spot check; full run pending)

`|corr(rank·)|`, r=1, mean over ROIs:

| method | census \|corr\| | reading |
|---|---|---|
| auglab_default (image-driven baseline) | **0.52** | texture preserved |
| **PALETTE (ours)** | **0.44** | texture preserved |
| synthseg_em | **0.04** | at the 0 floor — texture destroyed |

**Categorical gap (~10×)** between image-driven and generative. PALETTE (0.44, *with* Voronoi) ≈
auglab (0.52, *no* Voronoi) → the Voronoi parcellation costs only ~0.08; it does **not** destroy
texture. (`synthseg_noem` still transferring at time of writing; expected at the floor too.)

### No-blur ablation (both sets reported)

Because census r=1 is sensitive to blur, we also generate a **no-blur** version of *every* method
(blur/resolution disabled symmetrically — `data/generated_noblur/`) and report it **alongside**
the with-blur (training-config) set. This isolates the *contrast transformation's* texture
preservation from the method-agnostic blur augmentation. The with-blur set is the **headline**
(it is what the model trains on); the no-blur set is a labeled **mechanism ablation**. Expectation:
image-driven methods rise in the no-blur set (blur removed) while SynthSeg stays on the 0 floor
(it has no source texture to blur) — confirming that (a) blur, not the parcellation, is the main
contributor to PALETTE's moderate with-blur absolute, and (b) the categorical gap is invariant to
blur. Analysis runs over both sets in one table (`set` = blur / noblur / ref).

---

## Honest caveats (state these in the paper)

- **Absolutes are moderate and read relatively.** census r=1 is locally stringent — it punishes
  any blur/noise/resampling, so even the accepted AugLab baseline scores 0.52, not ~1. **No
  input-space metric puts PALETTE near 1.0**, because real PALETTE genuinely restructures contrast
  piecewise and applies mild blur (σ list `[0,0,0,0.3,0.5,0.8]`). The defensible claim is the
  categorical gap (image-driven ≫ SynthSeg ≈ 0), robust across metrics and r.
- **The phantom is for metric sanity only** (identity→1, gamma→1, noise→0), *not* for predicting
  PALETTE's value — with few random blocks it is high-variance and over-penalizes fragmentation.
  Trust the real data.
- **Level-1 is correlational supporting evidence.** It shows PALETTE preserves more source
  structure than SynthSeg — not that texture preservation *causes* the downstream win. The causal
  claim rests on **Level-3** (the noise-fill ablation with the partition held fixed).

---

## Parameter / design choices — justification (none tuned per result)

- **Window radius r ∈ {1,2} (reported grid).** Ordinal-neighbourhood size (as in LBP radius). We
  report both rather than fixing one; the categorical result holds across r. Not selected to
  maximize the gap.
- **Per-ROI over the 31 anatomical labels.** Localizes preservation to tissue classes (ties to
  the crush-case analysis). Standard practice; stated, not cited.
- **Eroded ROI masks (1 voxel, `binary_erosion`).** Excludes anatomical tissue borders shared by
  all methods.
- **64 MI bins.** Standard intensity-MI resolution; keeps the per-ROI 64×64 joint histogram
  populated (ROIs ≥ ~10³ voxels). NMI robust to bin count in 32–256.
- **Strict `>` for ties.** Flat regions → near-zero correlation, which is the correct answer
  (no texture), not an artifact.

---

## Is this fully grounded / not hacked? — honest answer

- **Grounded building blocks, used as intended:** rank/census transform (Zabih & Woodfill 1994,
  introduced *for* intensity-robust correlation); LBP family (Ojala 2002); NMI (Maes 1997 /
  Studholme 1999). **Not** a single verbatim published metric end-to-end — the exact composition
  (normalized rank, `|corr|` for inversion, per-ROI, eroded, r-grid) is documented standard-
  practice choices. We do **not** claim "100% literature-grounded end-to-end" (no real method is).
- **Not hacked:** the metric was chosen on *principle* (contrast+inversion invariance, texture
  literature) and validated on *controls* (identity→1, gamma→1, inverted→1, noise→0) **before**
  any real PALETTE/SynthSeg volume was seen; the real-data spot check then confirmed the predicted
  direction. Free parameters (r) are reported as a grid, not selected.

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

**Primary (adopted metrics):**
- Rank/census transform — Zabih & Woodfill 1994, *Non-parametric local transforms for computing
  visual correspondence*, ECCV.
- Local Binary Patterns (same ordinal family) — Ojala et al. 2002, IEEE TPAMI.
- MI / NMI for multimodal similarity — Maes et al. 1997, IEEE TMI; Studholme et al. 1999,
  Pattern Recognition.

**Considered, unsuitable here (contrast/inversion-sensitive):**
- GLCM/Haralick — Haralick et al. 1973, IEEE TSMC.
- Gabor / wavelet texture energy; FSIM, CW-SSIM, DISTS, STSIM.

**Landscape / framing:**
- MR image-to-image translation metrics — https://www.nature.com/articles/s41598-025-87358-0
- DL harmonization of structural MRI (structure-preservation via MI/SSIM/PCC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11365220/
