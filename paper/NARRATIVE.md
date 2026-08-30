# PALETTE — CVPR Narrative & Strategic Choices

Working notes on how we frame the paper. Decisions locked in discussion (2026-07-04).
Target: CVPR (8 pages), then MICCAI as fallback.

---

## 1. Method name

**PALETTE** — *Partition-based Affine Local-intEnsity Transform for Texture-prEserving augmentation.*

- Chosen over PRISM/TINT for being poetic + memorable; the metaphor *is* the method
  ("keep the drawing, swap the palette" = preserve structure, randomize contrast).
- Known caveat: collides with "Palette" (Saharia et al. 2022, image-to-image diffusion).
  Different sub-field; accepted.

---

## 2. What the method is (and its boundary)

- **Ours = AugLab (existing) + the PALETTE contrast transform (our contribution).**
- AugLab (NeuroPoly) = existing GPU augmentation library + nnU-Net trainer (strong
  intensity/spatial/artifact augs). This is the scaffold we plug into.
- **PALETTE transform** (`ImageContrastV26_6_2`, applied on-the-fly to the *real* image):
  1. 1-D K-means on foreground intensities → 2–6 intensity classes (label-free);
  2. Voronoi spatial sub-parcellation of each class (label-free — no atlas/template);
  3. signed-α affine remap per sub-region `y = μ + α(x−mean)`, α ∈ ±U(0.5, 2.0)
     (negative α → local contrast inversion);
  4. per-label affine decoupling (50%/label, the `_2` step) using GT seg.
- **Core invariant:** it remaps intensities of the real volume, so texture, gradients,
  partial-volume detail and unlabeled anatomy survive — only contrast is randomized.
- Selling points: novel operator, **plug-and-play**, near-zero latency, **one config
  everywhere** (tuned once on CHAOS, reused on all datasets — no per-dataset tuning).

---

## 3. Central thesis

> [Superseded 2026-08-02 — see Addendum below for the live framing.] A simple, label-free,
> **texture-preserving** contrast augmentation trains segmentation models that are robust
> across MRI contrasts (and modalities), reaching **new SOTA** — and we **explain** the SOTA
> via two orthogonal analyses: (1) a causal ablation (partition held fixed, only the fill
> changes) that isolates texture preservation as the active ingredient, and (2) an
> independent NGF texture-fidelity measurement confirming what the ablation predicts.

**The differentiator vs. SynthSeg:** *image-driven* (texture-preserving) vs.
*label-generative* (texture-destroying, GMM noise per label). This is the paper's reason
to exist. Note: SynthSeg's label-free (EM) variant has **no public code** — our
re-implementation is a fair baseline and minor contribution in itself.

**Pillar 2 (histogram-manifold coverage) is permanently out of scope for this paper** —
decided 2026-08-28: the honest result needs more justification than it's worth for CVPR, so
it will never be written up here, full stop. The original clause (2) above ("achieves the
most complete coverage of the real contrast-histogram manifold") is stale and has been
corrected in place rather than left to mislead a future read of this doc. Do not resurrect
it without a deliberate decision to reopen the analysis.

---

## 4. Framing decisions

- **Claim scope:** frame as *single-source, plug-and-play, texture-preserving augmentation
  for contrast/modality-robust segmentation.* This makes the natural comparison set
  SynthSeg + augmentation baselines, not multi-source DG methods.
- **"Beat SOTA," not "reach."** We outperform prior methods. But be honest about the
  *shape*: margin is **small on easy whole-organ Dice (+~1)**, **large on fine /
  texture-defined structures, exotic contrasts, and the failure tail.**
- **Lead with the honest aggregate, THEN drill into the large gaps.** Owning the thin
  average buys credibility to foreground the crush cases.
- **Foreground the "crush" cases** (modalities / regions where we dominate) — this is the
  effect-size story that a thin average can't carry. **Condition:** the crush cases must
  be *predicted a-priori by the texture principle* (texture-defined / low-contrast /
  most-shifted modality), never a hand-picked list. Report them with full rigor (all
  structures shown, per-case stats) so it can't read as cherry-picking.

---

## 5. Positioning vs. CVPR competition

These are **related, different setting** — cite + characterize crisply, do NOT ignore,
do NOT pretend they don't exist:
- **MADGNet** (CVPR 2024) — modality-agnostic DG, 6 modalities / 15 datasets; multi-source
  DG + custom architecture (vs our single-source, plug-and-play aug).
- **SynthFM** (CVPR 2025) — modality-agnostic foundation model.
- **Structure-Aware Stylized Image Synthesis** — conceptually adjacent (structure-preserving
  stylization); needs a crisp delta.
- **Direct comparison set we're held to:** SynthSeg / synthesis-based single-source family.

---

## 6. Experiments — table structure

- **Main comparison:** baseline (nnU-Net default aug), SynthSeg-EM, SynthSeg-noEM, **Ours**.
- **auglab_default is NOT in the main comparison** — it is literally `Ours − PALETTE`, so it
  belongs in the **ablation** (in the *main body*, stated explicitly as "Ours without the
  PALETTE transform"). Keeps the "we beat everything" narrative clean while staying
  transparent.
- **v26_6_2-alone (no AugLab) → ablation only.**
- **Statistical rigor:** paired per-case Wilcoxon signed-rank (n = cases, not folds) to
  neutralize the "it's noise / better tuning" objection. Necessary, not sufficient —
  significance kills "noise," not "marginal"; effect size comes from the crush cases.

### Datasets (canonical results in `datasets/{ds}/8_results_{ds}` ONLY — `eval/` is STALE)

**VERIFIED 2026-07-04 — Ours (auglab+PALETTE) is best on aggregate Dice in ALL 6
train-contrast settings.** Cross-fold, cross-class mean Dice, averaged over all test
contrasts/datasets ("all" column). Next-best is *always* `auglab_default` (= our ablation).

| Dataset | Base | **Ours** Dice-all | Next best | Margin | Ours HD95-all |
|---|---|---|---|---|---|
| on-harmony (brain, X-contrast) | T1w | **68.5** ✓ | 67.6 auglab_default | +0.9 | **4.2** ✓ |
| on-harmony | T2w | **67.8** ✓ | 67.4 auglab_default | +0.4 | **4.0** ✓ (tied) |
| CHAOS → AMOS/SLIVER07 (abd., X-modality) | T1in | **86.6** ✓ | 85.4 auglab_default | +1.2 | **27.4** ✓ |
| CHAOS → AMOS/SLIVER07 | T2spir | **86.0** ✓ | 84.4 auglab_default | +1.6 | **29.3** ✓ |
| BraTS-2024-glioma (brain tumor) | T1n | **46.6** ✓ | 46.2 auglab_default | +0.4 | 15.3 (auglab 15.1) |
| BraTS-2024-glioma | T2w | **47.0** ✓ | 46.6 auglab_default | +0.4 | 15.0 (auglab 14.9) |

**What the verified numbers confirm / qualify:**
- ✅ **Consistency holds: Ours wins aggregate Dice 6/6.** Strongest single argument against
  "noise/tuning" — same one config, wins across brain/abdomen/tumor, MRI + CT, two bases.
- ⚠️ **Margins are consistently thin (+0.4 to +1.6 Dice), and the runner-up is always our
  own ablation (`auglab_default`).** On BraTS the Dice win is a near-tie (+0.4) and HD95
  marginally favors auglab_default. → **This is exactly why the paper cannot rest on
  aggregate Dice; the crush-case + texture-mechanism story (§4, §7) is load-bearing, not
  optional.**
- ✅ **SynthSeg-noEM collapses on abdomen + tumor** (Dice 8.7 BraTS-T2w, 57.8 CHAOS-T2spir) —
  label-free-without-EM fails badly; **SynthSeg-EM is the real synthesis competitor**
  (~84–85 CHAOS, ~43 BraTS) and we beat it by +1.4 to +3.6.
- ✅ **Modality-agnostic headline holds under BOTH bases:** single MRI contrast in →
  CT liver **91.5** (T1in→SLIVER07) / **89.7** (T2spir→SLIVER07) Dice; AMOS-CT 76.7–78.3.
- ✅ **baseline collapses cross-contrast everywhere** (Dice-all 18–37), confirming the
  augmentation is doing the work.

**Notes on dataset roles:**
- **AMOS & SLIVER07 are cross-dataset TEST sets** (the CHAOS-trained model evaluated on them),
  not separate training experiments — their numbers live in the CHAOS cross-dataset tables.
- **trusted** = the CT-context + kidney-US "chimera" experiment — **rejected** (pandora box /
  attack surface); not in the paper.
- **open-ms** (brain MS lesions — the decisive small/texture-defined lesion test; launched
  2026-06-30) — results PENDING; expect texture advantage to grow FLAIR→T2W→T1W.
- Future healthy-spine benchmark (later).
- PET: dropped for now (organ-seg PET data scarce; would strengthen "modality-agnostic").

---

## 7. Proving texture superiority (THE gate for CVPR)

Texture preservation is currently a *mechanism guess*. Proving it is the single biggest
lever on acceptance. Three-level plan:

- **Level 1 — input-space, network-free (cheap, ~⅓ page).** Measure **recall of real
  texture**: of the source image's edges / high-freq energy, how much survives in the
  synthetic image. PALETTE ≈ high (affine is edge-preserving within region); SynthSeg ≈ 0.
  Use *recall of real texture* (not symmetric similarity) so Voronoi boundary artifacts
  (which *add* false edges, not remove real ones) don't confound it. Parallels the existing
  histogram/manifold analysis.
- **Level 3 — causal ablation (CENTERPIECE).** `PALETTE-structure + SynthSeg-fill`: identical
  K-means partition + (μ,α) distribution, but fill each region with GMM noise instead of
  remapping real intensities. One variable changes (texture preservation). If texture-defined
  structures collapse while boundary-defined ones don't, texture-preservation is *proven
  causal*, controlling for the parcellation. Direct PALETTE-vs-SynthSeg alone is *confounded*
  (differs in partition + fill + blur) → shows *what*, not *why*.
- **Level 2 — feature maps: DROPPED.** Too fuzzy / hand-wavy for the page budget.

Level 1 + Level 3 are complementary: L1 proves the *input* is texture-preserving (and
verifies the L3 ablation actually destroys texture); L3 proves texture-preservation *causes*
the win. If forced to one: **Level 3.**

---

## 8. Realistic odds (honest, no spin)

- **As-is** (thin margins, texture unproven): **~10–15%** at CVPR.
- **With decisive texture proof (L3) + crush-case framing + clean positioning vs incumbents:**
  **~30%.** A real shot, still uphill.
- **Regardless:** a strong MICCAI/MedIA paper today. The texture-proof work strengthens both
  versions, so it's not wasted. Plan: submit CVPR → MICCAI.

---

## 9. Open TODOs

- [x] Pull BraTS-2024 + CHAOS-T2spir + on-harmony-T2w canonical tables — **DONE 2026-07-04**;
      "combined wins aggregate Dice everywhere" confirmed (6/6), margins thin (see §6).
- [ ] Spec Level-3 noise-fill ablation concretely (plug point in the transform; datasets:
      open-ms + one abdominal; a-priori "texture-defined vs boundary-defined" structure split).
- [ ] open-ms results when training completes (the one pending dataset).
- [ ] Latency measurement (deferred, but a selling point — near-zero overhead).
- [ ] Per-case Wilcoxon significance across all main-comparison datasets (needed given thin margins).
- [ ] Decide how to present BraTS (near-tie on Dice, HD95 favors ablation) — likely rely on
      per-structure/crush-case breakdown there rather than the aggregate.

---

# ADDENDUM 2026-08-02 — narrative reframe (mechanism-first) + the "why" paragraph

## A. Proposed reframe: lead with the mechanism, not the SOTA

**Current thesis (§3, 2026-07-04):** "texture-preserving augmentation → new SOTA, *and* we
explain the SOTA via two analyses."
**Proposed thesis:** "**Preserving real texture matters for some segmentation tasks and not
others — we identify which, and PALETTE is the augmentation that exploits it.**" SOTA numbers
become supporting evidence, not the headline.

**Why this is better for CVPR, in one sentence:** it changes the sign of our weakest number.

| | SOTA-first framing | mechanism-first framing |
|---|---|---|
| CHAOS ladder +1.07 Dice | an embarrassment ("barely works on organs") | **the control arm** that makes the thesis falsifiable |
| "significant in 6 of 8" | thin, marginal | secondary; the dissociation carries the paper |
| §5 discussion admitting "margins are thin by design" | a concession a reviewer can quote back | unnecessary — margins are no longer the claim |

CVPR reviewers do not reward medical-benchmark deltas; they reward a transferable insight.
"Texture preservation pays off exactly when the target is appearance-defined rather than
interface-defined, and here is a controlled dissociation" is a CVPR-shaped contribution.
"We beat AugLab by ~1 Dice on 8 medical settings" is not.

**Structural consequences if adopted:**
- The ablation ladder (rung 4→5 fill swap) moves from §4.x to the **teaser figure**. It is an
  *intervention* — partition held fixed, only the fill changes — which is stronger evidence
  than any benchmark table.
- open-ms **+7.70** vs CHAOS **+1.07** is the headline pair, with the NGF dose-response
  (`ngf_dose_response_openms_flair.png`) as the independent confirmation that texture is what
  turned on at that rung.
- Contribution list reorders: (1) the finding, (2) PALETTE as the method that exploits it,
  (3) the 8-setting evaluation as breadth evidence.
- **Requirement this creates:** the dissociation needs ≥2 boundary-defined tasks, not just
  CHAOS T1in. **RESOLVED 2026-08-02** — CHAOS T2spir (−1.14 Dice) and BraTS T1n (+7.22) both
  landed, so the dissociation is now 2 appearance-defined (+7.70, +7.22) vs 2 interface-defined
  (+1.07, −1.14), with inconsistent sign on the interface side. Spine would make it 2-vs-3 and
  is now the highest-value remaining experiment. See `PAPER_TODO_20260802.md`.

## B. The "why" paragraph — drop-in, cite as written

Verified 2026-08-02. Full citation audit, including what NOT to cite and why:
`datasets/00_commun_scripts/00_04_analysis/label_cue_importance/LITERATURE_REVIEW.md` §9–10.

> Preserving real intensity structure matters where the target is defined by tissue
> appearance, and not where it is defined by an anatomical interface. Liver, spleen and
> kidney are encapsulated organs bounded by fat planes — a genuine physical interface, which
> is why edge-following methods such as geodesic active contours have long succeeded on them
> [CHAOS/Kavur 2021; GAC liver PMC4283827]; an augmentation that scrambles internal texture
> while preserving that interface loses little. Diffuse glioma has no such interface: tumour
> cells infiltrate beyond both the contrast-enhancing margin and the peritumoral FLAIR
> abnormality [Meta-Radiology 2025; FLAIRectomy Brain Sci 2022; PMC8156976], so the target
> must be inferred from tissue appearance rather than followed along an edge. MS lesions are
> likewise defined by tissue signal characteristics rather than an anatomical border
> [Zhang 2008].

**Three traps this wording avoids — do not "simplify" them back in:**
1. **Never argue from "CHAOS is solved."** A model reaching Dice 0.95 could be using texture,
   shape or position. Performance shows the task is learnable, not that the boundary is an
   intensity edge. Argue from *which class of algorithm works*: geodesic active contours are
   edge-followers and only work if a gradient ridge exists. Nobody has ever segmented MS
   lesions or glioma sub-regions with one.
2. **Never infer "texture-defined" from rater disagreement.** Low inter-rater agreement is
   equally consistent with "the scanner never captured the lesion extent". It supports "no
   clear boundary" at most, and even then weakly.
3. **Never claim MS lesions lack clear boundaries.** Clinical radiology describes typical MS
   lesions as *well* demarcated; "ill-defined borders" is a documented red flag pointing AWAY
   from MS (toward PML/NMOSD/MOGAD) — Brain 2019;142(7):1858. A neuroradiologist reviewer
   would invert this claim. For MS argue *texture-informative* + *hard to delineate*, never
   *boundary-less*. Glioma and MS are NOT symmetric; do not lump them.

**Also do not use:** the widely quoted glioma "20% intra / 28% inter-rater" figure
(attribution to Mazzara 2004 could not be confirmed); "manual liver Dice 0.95" as an
inter-rater number (it is manual vs a reference standard).

## C. Held in reserve — the contrast objection

Expect: *"boundary clarity is a property of the sequence, not the disease — enhancing tumour
has a razor-sharp margin on T1c."* Two answers, neither needed in the main text:
1. Glioma infiltration is documented beyond BOTH the enhancing margin AND the FLAIR
   abnormality — no sequence bounds the disease. The sharp T1c margin is a blood–brain-barrier
   boundary, not a tumour boundary.
2. Our own per-contrast measurement (804 ROIs, `label_cue_importance/FINDINGS.md`): every
   pathology label sits at or below chance **in its own best contrast** (enhancing tumour on
   T1c 0.463, MS on FLAIR 0.461) while every organ stays high in **both** contrasts
   (0.620–0.921). Rebuttal material, not main-text material.

## D. Status of this narrative doc

§3 (central thesis) and §4 (framing) predate this addendum and are **superseded on the
SOTA-first point** if the reframe is adopted. §8's honest odds (10–15% as-is, ~30% with the
texture proof) were written 2026-07-04; the ablation + NGF dose-response now exist, so ~30% is
the live number, and the reframe is an argument for the upper end of it — not a guarantee.
