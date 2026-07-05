# open-ms Pillar-2 — Histogram Coverage on a Sparse-Annotation Dataset

Second dataset for the PALETTE coverage analysis (companion to
`datasets/on-harmony/7_analysis_on-harmony/histogram_coverage_lvl_1/`). Same **metrics**
(per-group Coverage, Naeem 2020; Vendi, Friedman & Dieng 2023 — see that directory's
`GROUNDING_AUDIT.md`), but a **different feature space**, forced by — and demonstrating — the
sparse-annotation setting.

## Why open-ms, and why 2 regions

on-harmony has a dense 31-class anatomical parcellation; **open-ms ships only sparse binary MS
lesion labels** (no anatomy). This is the point, not a limitation: **SynthSeg is label-generative
— it needs dense anatomy to synthesize; PALETTE is image-driven — it works from the image
regardless of annotation density.** open-ms is the setting that exposes that difference.

We deliberately **do not** run SynthSeg to invent anatomy. We use the labels open-ms actually has:

- **Feature = [lesion histogram, brain histogram]** — 2 regions × 64 bins = **128-dim**.
  - `lesion` = FLAIR `dseg` mask (binary; all 3 contrasts are co-registered, so it applies to each).
  - `brain` ("overall") = the co-registered brainmask.
  - One global p1–p99 normalisation over the brain preserves where lesion intensities sit relative
    to brain tissue (bright FLAIR lesions vs dark T1w lesions).
- This is the honest 2-region analog of on-harmony's 31-region histograms. Limited resolution, by
  the dataset's nature — stated plainly.

## Data (all co-registered, 154×240×240 per subject)

- 30 subjects × {FLAIR, T1w, T2w}; lesion `dseg` present for all 30; brainmask per subject.
- **Real reference:** all 3 contrasts (90 scans) — coverage per (contrast × scanner) group.
- **Synthetic sources:** FLAIR **and** T1w (analysed separately → `outputs/flair/`, `outputs/t1w/`).

## Pipeline

1. **Generate** (GPU, romane) — `generate_openms_volumes.py`: 4 methods
   (palette / synthseg_em / synthseg_noem / auglab_default) × {FLAIR, T1w} × 30 subj × 10 variants.
   The transform **seg is a 2-region composite** `{background:0, brain:1, lesion:2}` from
   brainmask + lesion — the same 2 regions as the feature, and faithful to what the open-ms model's
   augmentation sees in training. SynthSeg-style generation therefore fills brain-as-one-tissue +
   lesion (coarse — expected on a dataset without dense anatomy; that coarseness is the finding).
   Reuses the dataset-agnostic per-method `configs/` from the on-harmony texture pillar.
2. **Extract** (CPU) — `extract_lesion_overall_openms.py`: real (3 contrasts) + synth (per source),
   producing the 128-dim [lesion, brain] histograms. Validated on real: 90 scans, both regions
   populate across all contrasts (co-registered lesion mask).
3. **Coverage + Vendi + PCA/UMAP** (CPU) — reuses the on-harmony
   `compute_coverage_metrics.py` / `plot_coverage_metrics.py` / `plot_manifold.py` **unchanged**
   (region-agnostic). Orchestrated by `run_coverage_openms.sh`.

## Honest expectations

- The comparative claim is the same as on-harmony (Coverage per real regime; PALETTE vs SynthSeg vs
  auglab), and it is **supporting/correlational**, not causal (the causal step is the Level-3
  training ablation). With only 2 regions the feature is coarse — read it as a sanity check that the
  on-harmony finding is not parcellation-specific, plus the qualitative sparse-annotation point.
- The SynthSeg arm is coarse **by construction** here (no dense anatomy). If it posts poor coverage,
  that is the expected consequence of its label-dependence — not a tuning artefact.

Grounding is inherited from the on-harmony `GROUNDING_AUDIT.md` (metrics identical); the only
open-ms-specific choice is the 2-region [lesion, brain] feature, justified above.
