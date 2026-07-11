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
tools (see next section).

**open-ms ROIs.** open-ms provides no anatomical parcellation — only a sparse binary MS-lesion
`dseg` (its brainmask is a computed extraction, not an annotation, so it is not used). We
therefore report **2 ROIs**: `lesion` (the FLAIR dseg, co-registered → applies to every contrast)
and `foreground` (intensity > 10% of the volume's p99 — a plain threshold stand-in for "brain",
**not** whole-image, since census on a mostly-background image is confounded by background–
background correlation). This is the same annotation-faithful ROI choice as the open-ms coverage
(Pillar-2) analysis. (The original on-harmony run used 31 anatomical labels + 1-voxel erosion; on
open-ms lesions are small/sparse, so erosion is off by default to avoid erasing them.)

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

### Also reported — `census_local8` (cancellation-robust, windowed variant)
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

**Honest status:** every ingredient is published and used as intended (SSIM structure term; census/
rank transform; local windowing), but — exactly like census_r1 — the end-to-end composition
(rank-transform → SSIM-structure-term/windowed-|corr| → averaged) is **not a single named published
metric**. The lone non-standard step is the `|·|` (SSIM's structure term is signed); we take it for
contrast-inversion invariance and state so. Report it as *"the SSIM structure component on census-
transformed volumes,"* not as an off-the-shelf named index.

*Why it is needed here.* Whole-ROI `|corr|` sums *signed* products across the entire ROI **before**
`|·|`. PALETTE's signed per-region α (`fromSeg.py`) makes some regions +correlated and others
−correlated with the source; over a large ROI these **cancel** in the sum, so census_r1
under-scores PALETTE for doing exactly what it is designed to do (randomize contrast direction
per region). Windowing confines each correlation to ≈one region, so the `|·|` is applied per block
and the signs no longer cancel. Use census_r1 for the **texture-vs-texture-blind** axis (clean, low
floor); use census_local8 for **PALETTE-vs-conventional-augmentation**, where census_r1 is
confounded. Caveat: smaller blocks have fewer samples → a **higher chance-correlation floor**
(~0.05–0.15 vs ~0.007 whole-ROI), compressing the bottom of the range. See `FINDINGS.md` §Anomaly 2.

### Dropped for open-ms — Normalized Mutual Information
`NMI = (H(X)+H(Y))/H(X,Y) ∈ [1,2]` (Maes 1997; Studholme 1999) was the on-harmony "content"
secondary metric but is **removed from the open-ms analysis**: it is computed on the 1-D set of ROI
intensity *values* (ignoring their spatial arrangement), so it is **spatially blind — not a texture
metric**. It reflects intensity-mapping determinism, not local structure preservation, and was
briefly (and wrongly) used to argue a PALETTE>auglab ordering; that argument was withdrawn. NMI
remains only in the shared on-harmony pipeline. (NGF and whole-ROI LNCC were also evaluated and
dropped earlier — NGF's η edge-parameter muddied interpretation.)

---

## Results

Live open-ms numbers (30 subjects × 10 variants/method; palette, synthseg_em, synthseg_noem,
auglab_default, v26_6_2_noisefill_v2; FLAIR + T1w sources; blur and noblur sets) are in
**`FINDINGS.md`** and the generated tables (`outputs/tables/summary.md`, `per_roi_*.csv`) — not
duplicated here so they don't go stale. Headline shape (noblur, census_r1, per ROI):

- **Texture-vs-texture-blind (the clean, decisive result):** palette (0.28–0.34) ≫ synthseg_em
  (0.09–0.15) > synthseg_noem ≈ v26_6_2_noisefill_v2 (floor ~0.05). `noisefill_v2` (PALETTE's
  partition with the real fill replaced by noise) collapsing to the floor is the causal control:
  the **fill** carries the texture, not the partition geometry.
- **PALETTE vs auglab_default:** on census_r1 auglab appears higher, but that is the sign-
  cancellation confound (§Anomaly 2, FINDINGS.md); under the cancellation-robust **census_local8**
  they are ~tied (lesion 0.58 ≈ 0.58; foreground 0.51 vs 0.62). No honest metric ranks PALETTE
  clearly above auglab on texture here — auglab_default in noblur is dominated by monotone,
  rank-preserving ops that genuinely preserve local structure. The categorical PALETTE ≫ SynthSeg
  gap is the paper claim; PALETTE-vs-conventional-aug is a wash on texture.

**What census actually measures:** not "what % of texture survived", but **whether the synthetic
image's local structure is *derived from the real anatomy* (correlated) or *invented* (independent
of it)**. Image-driven methods are strongly derived; SynthSeg is statistically independent (pure
fabrication) — real texture cues to learn vs none, the distinction that matters for a segmenter.

**Two SynthSeg subtleties resolved (see FINDINGS.md):** (1) SynthSeg's generator applies a
mandatory σ=0.5 blur even at native resolution (`functional.py:551`) that our `data_res`/
`blur_range` "noblur" knobs did NOT disable — the real switch is `apply_resolution:false`; the
blur manufactured a shared-boundary census that vanished once disabled. (2) synthseg_em stays
mildly above the noise floor because its EM step sub-partitions on the *real* image intensities
(its partition is real-informed even though the fill is random) — a genuine, defensible finding,
not an artifact.

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
- **Per-ROI over {lesion, foreground}.** The only annotations open-ms provides (see The
  measurement problem). Localizes preservation to the meaningful MS-lesion region + a brain-ish
  foreground. (on-harmony used 31 anatomical labels.)
- **Erosion off by default on open-ms.** on-harmony eroded 1 voxel to drop shared tissue borders;
  open-ms lesions are small/sparse and a 3³ min-pool erosion can erase them, so `--erode-iters 0`.
- **census_local8 block size = 8³, min 64 in-ROI voxels/block.** Block small enough to sit within
  ≈one PALETTE region (kills sign cancellation) yet large enough that per-block |corr| isn't pure
  chance. Reported alongside census_r1, not instead of it.
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
  any real PALETTE/SynthSeg volume was seen; the full real-data run (84 subjects) then confirmed
  the predicted direction (p=1.71e-15). Free parameters (r) are reported as a grid, not selected.

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
- Local-window aggregation of a similarity index (grounds census_local8) — Wang et al. 2004,
  *Image quality assessment: from error visibility to structural similarity* (SSIM; mean over
  local windows), IEEE TIP; local normalized cross-correlation — Avants et al. 2008, *Symmetric
  diffeomorphic image registration* (ANTs/SyN), Med. Image Anal.
- MI / NMI (on-harmony only; dropped for open-ms) — Maes et al. 1997, IEEE TMI; Studholme et al.
  1999, Pattern Recognition.

**Considered, unsuitable here (contrast/inversion-sensitive):**
- GLCM/Haralick — Haralick et al. 1973, IEEE TSMC.
- Gabor / wavelet texture energy; FSIM, CW-SSIM, DISTS, STSIM.

**Landscape / framing:**
- MR image-to-image translation metrics — https://www.nature.com/articles/s41598-025-87358-0
- DL harmonization of structural MRI (structure-preservation via MI/SSIM/PCC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11365220/
