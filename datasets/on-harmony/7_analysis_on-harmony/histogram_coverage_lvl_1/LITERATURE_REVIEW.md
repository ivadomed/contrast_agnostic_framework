# Histogram-Manifold Coverage — Literature Review & Metric Grounding

> **⚠ UPDATE 2026-07-05 — design finalised after discussion; parts of the body below are superseded.**
> The literature *landscape* here still holds, but three specifics changed (source of truth =
> `GROUNDING_AUDIT.md` §2.4–2.5):
> 1. **Feature space:** 7-region `regional_hist_64` → **31-class regional histograms** (1984-dim), matching
>    Pillar 1 (`Dataset031`). Real parcellated via each scan's **own per-modality synthseg → FreeSurfer→31**
>    (old code wrongly used the T1w seg for all modalities).
> 2. **Metric set:** PRDC-all-four → **Coverage only** (Naeem 2020), **per (scanner×contrast) group at the
>    cluster's local k-NN scale → macro-averaged**, + **Vendi** (global). Precision/Recall/Density dropped.
> 3. **Reference:** pooled-real-with-balancing → **per-group + macro-average** (balancing removed; it was the
>    crude version of what per-group does correctly).


**Purpose.** This is the second explanatory pillar of the PALETTE paper. Pillar 1 (texture,
`../texture_analysis_lvl_1/`) shows PALETTE *preserves* real anatomical texture where SynthSeg destroys
it. Pillar 2 (this analysis) asks a distributional question: **when we synthesise augmentations from T1w,
how well does each method's cloud of images cover the real MRI intensity-distribution manifold — and how
does that coverage decompose into fidelity vs diversity?**

This document establishes *grounding* before any experiment is run. Every headline number will map to a
named, citable metric. Nothing hand-rolled ships in the paper.

**Status:** grounding draft — no experiment run yet. Written 2026-07-05.

---

## 0. What we are NOT doing (lessons from the old analysis)

The pre-existing `../contrast_manifold/` pipeline carried the coverage idea but is unusable for the paper
as-is, for three reasons:

1. **Custom, unciteable geometry.** `plot_coverage.py` reports `spread_ratio`, `hull_coverage_2d`,
   `recall@Nx-scale`, `precision@Nx-scale`, and an `IND/OOD` split at P95 of real NN distance. These are
   *intuitive reinventions* of published metrics (e.g. `recall@Nx` ≈ Coverage; `spread_ratio` ≈ a
   diversity proxy) with no citation behind them. A reviewer will ask "why not cite the real metric?" —
   and they would be right. **Dropped entirely.** (Decision: PRDC + Vendi only, each cited.)

2. **Inverted framing.** Every interpretation doc there
   (`histogram_vs_curia_interpretation.md`, `v22_2_lhc_interpretation.md`) treats wide coverage as a
   *failure* ("the generator is contrast-overconfident", "spread ratio 3× real = bad"). That was the
   right read for a *harmonisation* generator. For a *domain-randomisation augmenter* the sign flips:
   spanning far beyond real clusters is the point. The metric layer is reusable; the narrative is rebuilt
   from scratch, and honestly (§5).

3. **Stale data.** `contrast_manifold/outputs/data/` holds v19–v28 (old guidance generators, LHC sweeps,
   CURIA embeddings). None of it is the three paper methods. This analysis instead reuses the **exact
   volumes generated for the texture analysis** (`../texture_analysis_lvl_1/data/generated/`), so both
   pillars rest on identical images (§4).

---

## 1. The question, stated precisely

Fix the real reference **R** = all ON-Harmony scans (pooled: T1w, T2w, FLAIR, GRE, dwi, bold, epi ×
Siemens/GE/Philips). For each augmentation method **M** ∈ {PALETTE, SynthSeg, auglab_default}, generate a
cloud of synthetic images **S_M** *from T1w only* and represent every image as a point in a fixed
intensity-histogram feature space (§3). We ask, comparatively across M:

- **Fidelity** — do S_M points land where real images live? (Precision, Density)
- **Coverage** — does S_M *reach* the real manifold, i.e. is each real image near some synthetic one?
  (Coverage, Recall)
- **Diversity** — how many effective modes does S_M explore? (Vendi)

The paper's hypothesis is **comparative, not absolute**: PALETTE — by remapping *real* intensities rather
than filling regions with parametric noise — should both (a) span widely (high Vendi) **and** (b) stay
anchored to realistic per-region structure (high Coverage/Density), whereas SynthSeg's Gaussian fill
should spray widely but into unrealistic regions (high raw spread, low Coverage/Precision). §5 states what
would *falsify* this so we are not fishing for a foregone conclusion.

---

## 2. The metric family (all citable)

These are the canonical distributional metrics for generative models. All operate on two point sets (real
**R**, fake **S**) in a feature space, via k-nearest-neighbour manifold estimates.

| Metric | Paper | One-line definition | Reads as |
|---|---|---|---|
| **Precision** | Kynkäänniemi et al., *NeurIPS 2019* | fraction of **S** falling inside the k-NN manifold of **R** | realism / fidelity |
| **Recall** | Kynkäänniemi et al., *NeurIPS 2019* | fraction of **R** falling inside the k-NN manifold of **S** | manifold overlap |
| **Density** | Naeem et al., *ICML 2020* | avg # of real k-NN balls each fake lands in (outlier-robust Precision) | fidelity, robust |
| **Coverage** | Naeem et al., *ICML 2020* | fraction of **R** whose k-NN ball contains ≥1 fake | **does S tile R** — the core claim |
| **Vendi** | Friedman & Dieng, *TMLR 2023* | effective # of modes = exp(Shannon entropy of kernel eigenvalues) | diversity, reference-free |

Lineage / why these five and not others:

- **Sajjadi et al., *NeurIPS 2018*** ("Assessing Generative Models via Precision and Recall") introduced
  the precision/recall *decomposition* of distribution matching — the conceptual root. We cite it as the
  origin but use the later k-NN estimators, which are the field standard.
- **Kynkäänniemi et al. 2019** made P&R computable via explicit k-NN manifolds. Precision and Recall come
  from here.
- **Naeem et al. 2020** showed Kynkäänniemi's Precision/Recall are *not robust to outliers and modeling
  of the manifold* and proposed **Density** and **Coverage** as fixes. Coverage is exactly our claim
  ("is every real point reached by a fake?"), and it is the one metric of the five that is provably
  robust to fake outliers — important because domain randomisation *deliberately* produces far-flung
  fakes that would inflate a naive recall.
- **Vendi (Friedman & Dieng 2023)** measures diversity **without a reference set** (effective number of
  distinct modes, via the eigenspectrum of a sample-similarity kernel). It is the honest way to quantify
  "how much does M explore" without conflating exploration with realism.

The `prdc` PyPI package (already a project dependency, used by the old `plot_prdc.py`) computes
Precision/Recall/Density/Coverage in one call and is the reference implementation of Naeem 2020. Vendi is
`vendi_score.vendi.score_X`.

**Together these five decompose the claim cleanly:** fidelity (Precision, Density) × overlap (Recall) ×
tiling (Coverage) × exploration (Vendi). SynthSeg's predicted signature (wide but unrealistic) and
PALETTE's predicted signature (wide *and* realistic) differ on *exactly* these axes — the metrics can
confirm or refute the mechanism, not merely restate it.

---

## 3. Feature space (what "histogram" means here)

Primary: **`regional_hist_64`** (448-dim) — for each scan, the SynthSeg macro-parcellation (7 regions:
WM, cortical GM, CSF/ventricles, subcortical GM, cerebellum, brainstem, whole-brain) is resampled to the
scan and a 64-bin intensity histogram is computed *per region*, after a single global [p1, p99]
normalisation so the **between-region ordering** (WM>GM for T1w; CSF>WM for T2w; CSF-suppressed for
FLAIR) is preserved. Extractor: `../contrast_manifold/scripts/extract_features_regional_hist.py`
(reused unchanged).

Why this space (grounding the choice):

- It **is** the histogram claim. The paper argues about intensity-distribution coverage; a per-region
  intensity histogram is the most direct, interpretable representation of exactly that. We are not
  outsourcing the claim to a learned perceptual embedding.
- It is the **cross-version standard** for this project (per CLAUDE.md, `regional_hist_64` is "the primary
  feature space for cross-version comparison").
- **Honest deviation from convention, stated up front:** PRDC/Vendi are usually applied in a *learned*
  embedding (Inception/DINOv2 for natural images). Applying them in a *hand-designed* histogram space is a
  deliberate deviation — and here it is a *feature, not a bug*: the claim is specifically about
  intensity-distribution coverage, not perceptual coverage, so an interpretable histogram space is more
  faithful to the claim than an opaque embedding would be. We will state this explicitly in the paper.

Preprocessing before the metrics (documented, standard): PCA fit on **R**, retaining 90% variance, applied
to both **R** and **S**. This is the same 0.90 setting the old `plot_prdc.py` used; it de-noises the raw
448-dim space and keeps k-NN distances meaningful (PRDC degrades in very high dimension). PCA is fit on
real only so the reference geometry is not contaminated by synthetic spread.

Secondary / robustness (report if primary is clean, do not lead with): `histogram_256` (global intensity
histogram) as a parcellation-free control, to show the result is not an artefact of the SynthSeg
parcellation.

---

## 4. Experimental design

- **Reference R:** all real ON-Harmony (pooled, multi-contrast, multi-scanner) — decision confirmed. This
  tests whether image-driven synthesis spans the *whole* real manifold, matching the domain-randomisation
  framing. (A T1w-only reference is a possible follow-up control but not the headline.)
- **Methods S_M:** PALETTE, SynthSeg (report EM and no-EM variants; lead with the stronger/standard one),
  auglab_default — the same three arms as the texture analysis.
- **Volumes:** reuse `../texture_analysis_lvl_1/data/generated/{palette, synthseg_em, synthseg_noem,
  auglab_default}` — 84 subjects each, all synthesised from T1w. **Identical images to Pillar 1**, so the
  two pillars are perfectly consistent. (Confirmed present: 84 cases per method.)
- **k for PRDC:** k=5 (Naeem 2020's recommended default) when sample sizes allow — not the adaptive
  k≤3 the old per-cluster script was forced into by tiny groups. With pooled R (thousands) and the full
  synthetic pool, k=5 is safe and standard.
- **Metrics per method:** Precision, Recall, Density, Coverage (one `compute_prdc` call, k=5, on the
  PCA-projected features) + Vendi (on S_M features, cosine kernel). Global — **no IND/OOD split**.
- **Statistics:** because the three methods are generated from the *same* 84 source subjects, coverage/
  fidelity can be bootstrapped over subjects (resample subjects with replacement, recompute PRDC) to get
  CIs and a paired comparison PALETTE vs SynthSeg — mirroring the paired Wilcoxon used in Pillar 1. This
  keeps the two pillars statistically consistent.

New scripts (to be written *after* this grounding is approved), kept minimal and each mapping 1:1 to a
cited metric:
- `scripts/extract_regional_hist.sh` — thin `run_job` wrapper around the existing extractor for the 4
  generated method sets + real ON-Harmony.
- `scripts/compute_coverage_metrics.py` — PCA(0.90) on real, then `compute_prdc(k=5)` + Vendi per method,
  with subject-level bootstrap CIs. One CSV out. No custom geometry.
- `scripts/plot_coverage_metrics.py` — grouped bar chart (5 metrics × 3 methods, CIs) + the paired
  bootstrap comparison. Reference lines only where a metric has a principled reference (Precision/Recall/
  Coverage ∈ [0,1]).

---

## 5. Honesty ledger — what could falsify the hypothesis, and what we will report either way

This is the section a skeptical reviewer (and the user) should read first.

1. **PALETTE might NOT dominate on Coverage of the *pooled* real manifold.** The old analysis found
   near-zero coverage of true T2w/FLAIR/DWI clusters for *every* image-driven method — no T1w-derived
   synthesis reproduces real T2w/FLAIR physics. So absolute Coverage against pooled-real may be modest for
   **all three** methods. The defensible claim is therefore **comparative** ("PALETTE covers more of the
   real manifold than SynthSeg, at higher Density/Precision"), not absolute ("PALETTE covers the real
   manifold"). We will report absolute numbers plainly and let the comparison carry the argument.

2. **SynthSeg could win on raw diversity (Vendi) or Recall.** Gaussian fill with random per-region means
   spans an enormous intensity volume; it may well post the highest Vendi. That is *not* a loss for our
   story — it is the point: high diversity with **low Precision/Density/Coverage** is precisely the
   "sprays into unrealistic regions" signature. If SynthSeg instead posts high Vendi **and** high
   Precision/Coverage, the mechanism hypothesis is *weakened* and we will say so.

3. **The metrics decompose the claim; they do not by themselves prove it drives segmentation accuracy.**
   Like Pillar 1, this is **supporting/correlational** evidence about the *input distribution*. The causal
   link to downstream Dice is the job of the Level-3 training ablation (PALETTE vs PALETTE-noise-fill,
   already launched). We will not overclaim causality from coverage numbers.

4. **Feature-space dependence.** PRDC values depend on the feature space and PCA cut. We fix both a priori
   (regional_hist_64, PCA 0.90, k=5) — no post-hoc sweeping of these knobs to favour a result (that would
   be p-hacking, the exact trap flagged in Pillar 1). The `histogram_256` control (§3) guards against a
   parcellation artefact.

5. **Coverage is asymmetric.** Coverage/Density/Precision depend on which set is "real". R = ON-Harmony,
   S = synthetic, fixed for all methods — the only correct orientation for "does synthetic cover real".

**Bottom line for grounding:** every headline number is Precision / Recall / Density / Coverage
(Kynkäänniemi 2019; Naeem 2020) or Vendi (Friedman & Dieng 2023), computed with fixed a-priori
hyperparameters in the interpretable histogram space that *is* the claim, on the same volumes as the
texture pillar, with a comparative hypothesis and an explicit falsification condition. No hand-rolled
metric survives into the paper.

---

## 6. References

- Sajjadi, Bachem, Lucic, Bousquet, Gelly. "Assessing Generative Models via Precision and Recall."
  *NeurIPS 2018.*
- Kynkäänniemi, Karras, Laine, Lehtinen, Aila. "Improved Precision and Recall Metric for Assessing
  Generative Models." *NeurIPS 2019.*
- Naeem, Oh, Uh, Choi, Yoo. "Reliable Fidelity and Diversity Metrics for Generative Models." *ICML 2020.*
  (Density & Coverage; the `prdc` reference implementation.)
- Friedman, Dieng. "The Vendi Score: A Diversity Evaluation Metric for Machine Learning." *TMLR 2023.*
