# Label boundary-cue importance — findings (2026-08-02)

What local image evidence marks a segmentation label's surface, and how does that differ between
organ labels (CHAOS) and pathology labels (BraTS tumour sub-regions, open-ms MS lesions)?

Method, citations and every honesty statement: `LITERATURE_REVIEW.md` (same directory).
Metric definitions: `cue_metrics.py`. Chance level for every AP column is **0.500** (positives and
negatives are balanced by construction); for every z column it is **0**.

**Run that produced these numbers:** TamIA job 392932, 40 BraTS subjects (fixed seed) x 4
modalities, all 22 open-ms, all 16 CHAOS; 804 (subject, label) files; all volumes resampled to
**1 mm isotropic** first (see §3.1 — this was a bug fix, not a cosmetic choice).

---

## 1. Headline: a double dissociation

| dataset | `eg_coh` (is there a wall) | `bg` (intensity difference) | regional HOG z, σ=2 (is the region texturally distinct) |
|---|---:|---:|---:|
| **CHAOS** (organs) | **0.776** | 0.810 | **−1.4** (at/below null) |
| **BraTS** (tumour sub-regions) | 0.477 | 0.591 | **+3.7** |
| **open-ms** (MS lesions) | 0.443 | 0.538 | **+7.3** |

> Organ boundaries are marked by a strong coherent intensity wall and carry **no** distinctive
> regional texture. Tumour and lesion boundaries have **no such wall** — at or below the chance
> line — but their regions **are** texturally distinct from surrounding tissue, increasingly so at
> coarser scales.

Two independent measurement families point opposite ways for the two groups. That is a
dissociation, not one axis on which a group happens to score lower.

**Scale dependence (regional HOG z, dataset means):**

| dataset | σ=0 | σ=1 | σ=2 |
|---|---:|---:|---:|
| CHAOS | −1.7 | −1.6 | −1.4 |
| BraTS | +0.4 | +0.9 | +3.7 |
| open-ms | +1.5 | +4.3 | +7.3 |

The texture effect **grows monotonically with scale** for both pathology datasets. This is why the
local half-ball texture cue (`tg`, radius 7, unsmoothed) reads at chance: it is measuring at the
wrong scale, and it measures a different thing (a texture *step at the outline* rather than
regional *distinctiveness*). σ=4 is mostly `nan` — the 3σ exclusion band leaves too few voxels —
so the trend is established only out to σ=2.

## 2. Per-label detail (selected; full table in `cue_importance.csv`)

| label | n | bg | eg_coh | eg_sharp | tg | HOG z σ=2 | NGF z |
|---|--:|--:|--:|--:|--:|--:|--:|
| CHAOS t2spir right_kidney | 16 | 0.918 | **0.921** | 0.381 | 0.557 | −1.3 | +10.6 |
| CHAOS t2spir left_kidney | 16 | 0.900 | **0.885** | 0.383 | 0.554 | −1.8 | +11.5 |
| CHAOS t2spir spleen | 16 | 0.896 | **0.865** | 0.381 | 0.557 | −0.4 | +12.2 |
| CHAOS t2spir liver | 16 | 0.806 | 0.728 | 0.420 | 0.538 | −1.7 | +10.5 |
| CHAOS t1in liver | 16 | 0.721 | 0.635 | 0.434 | 0.551 | −1.4 | +9.4 |
| BraTS t2f whole_tumour | 40 | 0.756 | 0.585 | 0.454 | 0.609 | +2.3 | +9.3 |
| BraTS t2f SNFH | 40 | 0.759 | 0.547 | 0.425 | 0.571 | +1.5 | +10.2 |
| BraTS t1n NCR | 16 | 0.484 | **0.352** | 0.471 | 0.400 | +5.5 | +7.4 |
| BraTS t1n ET | 26 | 0.515 | **0.398** | 0.498 | 0.416 | +5.5 | +3.8 |
| open-ms flair lesion | 22 | 0.581 | 0.461 | 0.399 | 0.518 | +2.2 | +16.7 |
| open-ms t1w lesion | 22 | 0.495 | 0.424 | 0.470 | 0.436 | **+12.4** | +7.3 |

`whole_tumour` on T2-FLAIR is the most boundary-visible BraTS target (`bg` 0.756, `eg_coh` 0.585) —
consistent with WT on FLAIR being the one BraTS label that genuinely is fairly edge-defined. Worth
reporting rather than hiding: it pre-empts the obvious objection.

## 3. Corrections made during this study (each was caught by a control, not by eyeballing)

### 3.1 Voxel-vs-millimetre — invalidated the first cross-dataset comparison
Every radius was in **voxels**. CHAOS is 2.9×–6.6× anisotropic (1.36–1.89 mm in-plane, 5.5–9.0 mm
slices); BraTS/open-ms are 1 mm isotropic. So the "radius-7 ball" was a true 7 mm sphere in the
brain data and a ~10 × 10 × 50–63 mm **cigar** in CHAOS, pooling gradients across ten slices of
unrelated anatomy — destroying exactly the coherence `eg_coh` exists to measure. After resampling
all volumes to 1 mm isotropic, CHAOS `eg_coh` went **0.639 → 0.776** while BraTS (0.472 → 0.477)
and open-ms (0.441 → 0.443) barely moved — the signature of a real geometry bug, since only the
anisotropic dataset changed.

### 3.2 The regional null was mis-specified
First version moved the ROI to a **random location**. A randomly placed blob straddles several
tissue types, inflating its inner-vs-shell contrast, so the null came out *harder* than the real
case and every dataset scored z ≤ 0 — including CHAOS, where liver-vs-fat texture unquestionably
differs. Replaced with **shell-vs-shell at the same location** (split the surrounding shell by a
random plane), which controls for local heterogeneity instead of importing it.

### 3.3 Smoothed HOG leaked intensity steps
Blurring spreads the boundary's own gradient into both regions. A pure intensity step with zero
texture change scored z = +9.2 at σ=2 with a 2-voxel band, and still +4.8 with a 2σ band. Fixed
with a **3σ exclusion band**. Regions too small to survive the band return `nan` for that scale
rather than a contaminated number.

### 3.4 Other rejected components
- The ordinal extremum code was **removed from the regional measure**: it leaked intensity steps at
  regional scale and its null had near-zero spread, once producing z = +7345. It is sound as the
  local `tg` cue and is retained only there.
- Per-orientation NGF alignment `(∇I·u)²/(‖∇I‖²+ε²)` was **rejected as degenerate**: maximised over
  candidate orientations, cos² saturates near 1 everywhere (AP 0.571 on a clean step edge).
  Replaced by structure-tensor coherence.
- Plain LBP was **rejected**: it scored tg = 1.000 on a pure intensity *ramp* with no texture change.
- Range-normalised sharpness was **rejected**: it made pure noise look maximally sharp.
- z-scores were unstable at 10 null draws (a control read +3.6 on one seed, +1.8 on another);
  defaults raised to 40 nulls / 16 repeats, where controls hold within ±2.4 across seeds.

Final control panel (`cue_metrics.py --sanity`, all passing): a genuine texture difference reads
+6.0/+6.6/+4.3 across scales; identical texture reads −0.7/−0.6/+2.0; a pure intensity step reads
−0.2/+1.7/+2.0.

---

## 4. Reviewer attacks — ranked by how much damage they do

### A1. "Your CHAOS 'no texture' result is an upsampling artifact." — **STRONGEST, NOT YET REFUTED**
CHAOS was upsampled 5.5–9× along z by trilinear interpolation, which manufactures smooth,
texture-free data on that axis. That could by itself flatten the ROI-vs-shell HOG contrast and
produce the null result, with no anatomical meaning. Claim (a)'s cross-dataset contrast rests on
this row.
**Refutation to run:** take open-ms (isotropic, z = +7.3), simulate CHAOS anisotropy (decimate z by
6×, upsample back), re-measure. If the texture signal survives, the CHAOS null is anatomy; if it
collapses, the CHAOS null is interpolation. **This control has not been run.** Until it is, state
claim (a) as "BraTS and open-ms lesions carry regional texture distinctiveness" and treat the
CHAOS comparison as suggestive.

### A2. "eg_coh < 0.5 means your negatives are edge-rich, not that tumours lack walls." — **SERIOUS**
Negatives are non-boundary points in nearby tissue. In brain that tissue is *full* of unlabelled
boundaries (WM/GM, ventricles, sulci); in abdomen the tissue between organs is comparatively
homogeneous. So a sub-chance AP may reflect strong negatives rather than a weak boundary. AP is a
*relative* statistic and this is exactly where that bites.
**Refutation to run:** report the **absolute** mean `eg_coh` at positives vs negatives per dataset,
not only the AP. If tumour boundaries have genuinely low absolute coherence, the claim stands; if
they are moderate and the negatives are simply higher, the claim must be reworded.
**This check has not been run** (blocked: `/tmp` full + relay needs 2FA).

### A3. "Small lesions are dropped by your 3σ band, so you report only large confluent lesions."
The band erodes the ROI by 6 voxels at σ=2. Small MS lesions and small NCR/ET regions may not
survive, leaving a selection of large lesions — which are also the most texturally developed.
**Check not run** (same blocker): count `nan` per dataset per scale. If open-ms σ=2 rests on a
handful of large lesions, the +7.3 is about those, not about MS lesions generally.

### A4. "Local `tg` and regional HOG disagree — which is it?"
CHAOS scores *highest* on local `tg` (0.548) and *lowest* on regional HOG; the pathology datasets
do the reverse. This is defensible but must be stated crisply, not buried: they measure different
things. `tg` = "is there a texture **step exactly at the outline**" (a liver edge genuinely has
liver texture on one side and fat/bowel on the other). Regional HOG = "is the region's texture
**distinct from the tissue around it**". A gradually infiltrating tumour can be texturally
distinctive with no step at its drawn edge. If a reviewer reads the disagreement as instability
rather than as two questions, the paper has failed to explain it.

### A5. "Your foreground/surroundings are not comparable across datasets."
Brain data is skull-stripped, so "foreground" is brain; CHAOS is a whole abdomen, so foreground is
body. The shell around a liver contains other organs and fat; the shell around a tumour is brain.
The comparison is between different kinds of "surroundings". Inherent to the question, but a
reviewer will want it acknowledged.

### A6. "No multiple-comparison correction."
30 label rows × 7 statistics, uncorrected. Mitigated by the claim resting on dataset-level means
with consistent direction across all rows rather than on any single cell — but say so explicitly.

### A7. "This whole analysis has no published precedent."
There is no prior work doing cue-importance attribution for medical segmentation labels
(LITERATURE_REVIEW §7). The skeleton is strongly grounded (Martin/Fowlkes/Malik 2004 is a close
match) but every 3-D cue is a composition of cited parts validated by phantoms. Present it as a
methods contribution, not as "we applied a published metric".

### A8. "You measure annotation, not anatomy."
Positives are wherever the annotator drew. Published BraTS boundary variability is ~20% intra-rater
and ~28% inter-rater. Where annotators drew a line the image does not support, this analysis
faithfully reports "no evidence" — which is arguably the point, but the interpretation boundary
must be stated.

### A9. Smaller ones
- `eg_sharp` on CHAOS (0.413) is interpolation-contaminated; do not report it. Lead with `eg_coh`.
- σ=4 is mostly `nan`; the scale trend is established only to σ=2.
- HOG uses 13 discrete orientation bins — coarse, and residual anisotropy could bias which bins fill.
- For multifocal MS, one lesion's "surrounding shell" may contain other lesions, diluting contrast
  (this biases *against* the open-ms result, so it is conservative).
- Sample size per ROI differs greatly (organs ≫ lesions). Within a comparison all four histograms
  use equal-size subsamples and the null uses the same n, so z is internally controlled; across
  datasets, smaller n means a wider null and therefore a *smaller* z — again conservative for the
  lesion datasets.

## 5. Status

- **Claim (b) — pathology labels lack the sharp boundary organs have: SUPPORTED**, large and
  consistent across every row and two cue families, and made *stronger* by the geometry fix.
- **Claim (a) — pathology labels carry regional texture distinctiveness: SUPPORTED for BraTS and
  open-ms**; the *comparison against CHAOS* is pending control A1.
- Blocked checks A1–A3 should be run before this goes in a paper.
