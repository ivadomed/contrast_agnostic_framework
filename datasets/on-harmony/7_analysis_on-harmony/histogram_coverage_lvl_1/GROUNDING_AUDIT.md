# Grounding Audit — Histogram-Manifold Coverage (Pillar 2)

**Purpose.** A component-by-component check that *every* element of this experiment is either (a)
backed by a citable paper, (b) uncontroversial standard practice, or (c) an explicit design choice
that must be disclosed (not dressed up as "grounded"). Written 2026-07-05 as a deliberate triple-check.
Companion to `LITERATURE_REVIEW.md` (design rationale) — this file is the citation ledger.

**Verdict up front:** the *metrics and building blocks are fully citable*. The honesty burden is on two
remaining **choices** — the PCA-dimensionality cut and the use of a hand-designed histogram space instead
of a learned embedding. None is hidden; each is disclosed below and handled (the PCA cut with a sensitivity
sweep). Ungrounded metrics from the old pipeline (IND/OOD split, spread_ratio/hull/recall@Nx) were removed.

**Design finalised 2026-07-05 (after discussion — see below):** metric set reduced to **two axes, no
redundancy** — **Coverage** (Naeem 2020), computed **per (scanner×contrast) group → macro-averaged**, and
**Vendi** (global). **Precision, Recall, and Density were dropped** (§2.4), and the earlier **balancing of
the pooled reference was removed** — the per-group + macro-average design *is* the principled replacement
for balancing (each real regime weighted equally, at its own local k-NN scale, with no data discarded).

Legend — **[CITED]** load-bearing, from a paper · **[STD]** standard practice, no single canonical cite ·
**[CHOICE]** a decision we made; disclose, don't cite.

---

## 1. Component ledger

| # | Component | Role | Status | Source / justification |
|---|---|---|---|---|
| 1 | ~~Precision, Recall~~ **DROPPED** | fidelity, overlap | — | Kynkäänniemi 2019 — non-robust originals of Density/Coverage; Recall inflated by domain-randomisation outliers (§2.4). Not reported |
| 1b | ~~Density~~ **DROPPED** | fidelity | — | Naeem 2020 — fidelity is not our objective; global density is GRE-confounded and redundant with Coverage+Vendi here (§2.4). Not reported |
| 2 | **Coverage** (per group → macro-avg) | reach — does synth tile each real regime | **[CITED]** | Naeem et al., *Reliable Fidelity and Diversity Metrics*, ICML 2020. Per-group because real is a union of clusters, so the k-NN radius must be each cluster's local scale (§2.5) |
| 3 | `prdc` implementation | computes Coverage | **[CITED]** | Official Naeem-2020 package. **Verified in source** (§3): `coverage` = fraction of real with a fake in its k-NN ball; kth=`k+1` excludes self |
| 4 | k = 5 (nearest_k) | PRDC neighbourhood | **[CITED]** | Naeem 2020 experimental default (package default = 5) |
| 5 | **Vendi score** | diversity (reference-free) | **[CITED]** | Friedman & Dieng, *The Vendi Score*, TMLR 2023. **Verified in source**: L2-normalise rows → K=XXᵀ (cosine kernel) → order-1 (Shannon) Vendi |
| 6 | **31-class parcellation** | region definition | **[CITED]** | SynthSeg (Billot et al., *Medical Image Analysis* 2023) labels; the 31 lateralised classes are a FreeSurfer *aseg* subset (Fischl et al., *Neuron* 2002). Identical to Pillar 1 & to `Dataset031` |
| 7 | **FreeSurfer→31 map** | real multi-contrast labels | **[CITED]** | Standard FreeSurfer *aseg* LUT (Fischl 2002). **Verified**: complete bijection, 31 entries → 1..31 once; matches `Dataset031/dataset.json` names (e.g. FS 2→WM_L, 41→WM_R, 16→Brainstem) |
| 8 | **PCA** (before PRDC & for viz) | dimensionality reduction | **[CITED]** | Hotelling 1933 / Jolliffe 2002. *Reducing dims before k-NN metrics* is motivated by distance concentration in high-D (Beyer et al. ICDT 1999; Aggarwal et al. ICDT 2001) — **[CITED]** for the rationale |
| 9 | **UMAP** (viz only) | 2-D manifold plot | **[CITED]** | McInnes, Healy, Melville 2018. *Qualitative only* — no quantitative claim rests on UMAP geometry |
| 10 | **Subject-level bootstrap** | CIs + paired p-value | **[CITED]** | Efron 1979; cluster/grouped resampling — Efron & Tibshirani 1993. Resampling the 84 subjects (not runs) respects non-independence of a subject's 10 runs |
| 11 | Intensity histogram (per region, 64 bins) | feature | **[STD]** | Histograms as image descriptors are textbook; MRI intensity-histogram representation cf. Nyúl & Udupa, *MRM* 1999. The specific **31-region concatenation** is a **[CHOICE]** (see §2.3) |
| 12 | p1–p99 global min-max within brain | normalisation | **[STD]** | Robust percentile clipping is standard MRI preprocessing. Rationale: one *global* scale preserves inter-region ordering, which *is* MRI contrast |
| 13 | Real fit for scaler + PCA (not combined) | reference geometry | **[CHOICE]** | Fit on real only so the feature space is fixed & identical across methods (else each method's synth shifts it → PRDC not comparable). Disclosed |
| 14 | PCA variance cut (0.5–0.9) | dims kept | **[CHOICE]** | Not from a paper. Handled by a **sensitivity sweep** — report the ranking across cuts, choose the operating point on distance-concentration grounds, never tune to a result |
| 15 | ~~Balance real by (modality×scanner)~~ **REMOVED** | reference composition | — | Superseded by per-group Coverage + macro-average (§2.5), which weights each regime equally at its own local scale without discarding data. Real subsampled to ≤300/group only as a compute bound |
| 16 | 64 bins; UMAP n_neighbors=30/min_dist=0.1; cosine kernel for Vendi | minor params | **[CHOICE]** | Conventional defaults; disclosed. None load-bearing for a headline number |

Removed from the old pipeline (were ungrounded): **IND/OOD P95 split**, **spread_ratio**, **hull_coverage_2d**, **recall@Nx-scale**, **precision@Nx-scale**. None had a citation.

---

## 2. The three disclosed choices (where a reviewer will push)

**2.1 PCA-dimensionality cut [#14].** k-NN metrics degrade under distance concentration in high dimension
(Beyer 1999; Aggarwal 2001), so PRDC is computed in a reduced PCA space — this *is* grounded. What is *not*
paper-fixed is the exact cut. Risk: choosing the cut after seeing results = p-hacking. Mitigation: a
**sweep over 0.5/0.6/0.7/0.8/0.9 variance**, reporting whether the method ranking is stable; the operating
point is justified by the concentration rationale, not by which cut flatters PALETTE. If the ranking flips
with dimension, we report that as the finding.

**2.2 Balancing the real reference — REMOVED, superseded by §2.5.** (Kept here for provenance.) An earlier
version balanced the pooled reference by stratified downsampling because real is ~81% GRE. Per-group
Coverage + macro-average is a strictly better fix (equal weight per regime, own local scale, no data
discarded), so balancing was dropped.

**2.3 Hand-designed histogram space, not a learned embedding [#11].** PRDC/Vendi are conventionally
computed in an Inception/DINO embedding. We use per-region intensity histograms. This is a *deliberate*
deviation and arguably *more* faithful here: the claim is specifically about intensity-distribution
(contrast) coverage, not perceptual coverage, so an interpretable histogram space matches the claim better
than an opaque embedding. Disclosed as a deviation, with a `histogram_256` (parcellation-free) control
available to rule out a parcellation artefact.

**2.4 Why only Coverage + Vendi (dropping Precision / Recall / Density).** Two axes suffice and each maps
to a distinct question: *reach* (Coverage) and *diversity* (Vendi). The others were dropped on principled,
not cosmetic, grounds:
- **Precision & Recall** are the *non-robust originals*; Naeem 2020 introduced Density & Coverage precisely
  as their outlier-robust replacements. Reporting both is redundant, and the non-robustness bites hardest
  here because domain randomisation *deliberately* produces outlier samples (we saw Recall inflate to 0.82
  via spread-engulfing — exactly the artefact Coverage corrects).
- **Density** is a *fidelity* axis ("does synth look real"), which is **not our objective** — low fidelity
  is the design working, not a defect, so a density number invites the wrong "bad generator" read. Computed
  globally it is also GRE-confounded (the same majority-domination §2.5 removes from Coverage), and it is
  largely redundant with what Coverage + Vendi already convey (exploratory vs conservative). Its only
  unique job — detecting "spray-bought" coverage — is not needed, because the **downstream Dice result
  settles utility causally**, independent of any fidelity metric.

**2.5 Per-group Coverage at the local scale [#2, replaces #15].** Real ON-Harmony is a *union of
well-separated (scanner×contrast) clusters*, not one manifold. Coverage draws a k-NN ball around each real
point; that ball's radius is a *local scale*. Pooled, the radius is set by the dense majority cluster (GRE),
so sparse clusters (FLAIR, 9 scans) are measured at the wrong scale and a pooled Coverage number collapses
to "coverage of GRE, contaminated at the edges." Computing Coverage **per group at its own scale, then
macro-averaging**, fixes this at the root and is the principled form of balancing. Groups with < k+1 real
are reported **N/A explicitly** (logged), never silently excluded — otherwise the macro-average would quietly
drop the hardest regimes. This restores the *right* core of the old `plot_prdc.py` (per-group PRDC) while
leaving out its ungrounded add-ons.

---

## 3. What was verified in source (not assumed)

- **`prdc.compute_prdc`** — read the installed source: `precision`/`recall` = fraction inside the other
  set's k-NN manifold (Kynkäänniemi); `density = (1/k)·Σ …` and `coverage` = fraction of real with a fake
  in its k-NN ball (Naeem); neighbour radius uses `k+1` to exclude self. Matches Naeem 2020 exactly.
- **`vendi.score_X`** — read the source: `normalize(rows) → K = X·Xᵀ → score_K(q=1)`, i.e. cosine-kernel,
  order-1 (Shannon) Vendi. Matches Friedman & Dieng.
- **FS→31 map** — asserted complete bijection (31 keys → {1..31}); label names cross-checked against
  `Dataset031/dataset.json` (WM_L…VentralDC_R) and the FreeSurfer aseg LUT.
- **Feature parity real↔synth** — both use the identical `compute_features_31` (same p1–p99 norm, same 64
  bins, same 31 regions); real uses each scan's own per-modality synthseg (FS→31), synth uses `Dataset031`
  labelsTr directly (already 1..31). The old code's bug — T1w synthseg applied to every modality — is fixed.

## 4. Known interpretive caveats (carried from LITERATURE_REVIEW.md §5)

- **Recall (Kynkäänniemi) is inflated by spread-out fakes** whose large k-NN balls engulf reals; **Coverage
  (Naeem) is the robust counterpart** and is the metric we trust for "tiles real." We report both and lead
  interpretation with Coverage.
- The claim is **comparative, not absolute**: no T1w-derived method reproduces real T2w/FLAIR/GRE physics,
  so absolute Coverage against pooled real is expected to be low for *all* methods.
- These are **input-distribution** metrics — supporting/correlational evidence, not causal. The causal step
  is the Level-3 training ablation.

## 5. Bottom line

Every headline number maps to a citable metric (rows #1–5) computed over a citable parcellation (#6–7) with
a citable reduction (#8) and citable uncertainty (#10). The load is on three disclosed choices (#13–15),
of which the PCA cut is the only one that could bias a result — and it is neutralised by the sweep. Nothing
hand-rolled or uncited survives into a claim.

## 6. References
- Sajjadi et al. "Assessing Generative Models via Precision and Recall." NeurIPS 2018.
- Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS 2019.
- Naeem et al. "Reliable Fidelity and Diversity Metrics for Generative Models." ICML 2020.
- Friedman & Dieng. "The Vendi Score." TMLR 2023.
- Billot et al. "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining." Medical Image Analysis 2023.
- Fischl et al. "Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain." Neuron 2002.
- McInnes, Healy, Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." 2018.
- Beyer et al. "When is 'nearest neighbor' meaningful?" ICDT 1999. · Aggarwal et al. "On the surprising behavior of distance metrics in high dimensional space." ICDT 2001.
- Efron. "Bootstrap Methods: Another Look at the Jackknife." Annals of Statistics 1979. · Efron & Tibshirani. "An Introduction to the Bootstrap." 1993.
- Nyúl & Udupa. "On standardizing the MR image intensity scale." Magnetic Resonance in Medicine 1999.
- Hotelling. "Analysis of a complex of statistical variables into principal components." 1933. · Jolliffe. "Principal Component Analysis." 2002.
