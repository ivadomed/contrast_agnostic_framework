# open-ms Pillar-2 — Histogram Coverage on a Sparse-Annotation Dataset

Second dataset for the PALETTE coverage analysis (companion to
`datasets/on-harmony/7_analysis_on-harmony/histogram_coverage_lvl_1/`). Same **metrics**
(per-group Coverage, Naeem 2020; Vendi, Friedman & Dieng 2023 — see that directory's
`GROUNDING_AUDIT.md`), but a **different, deliberately minimal feature space**, forced by — and
demonstrating — the sparse-annotation setting.

## Why open-ms, and why 2 histograms

on-harmony has a dense 31-class anatomical parcellation; **open-ms provides only sparse binary MS
lesion labels** (no anatomy). This is the point: **SynthSeg is label-generative — it needs anatomy to
synthesize; PALETTE is image-driven — it works from the image regardless of annotation density.**
open-ms is the setting that exposes that difference.

We use ONLY the labels open-ms actually annotates (its brainmask is a computed brain-extraction, **not**
an annotation, so we do not use it):

- **Feature = two histograms** (2 regions × 64 bins = 128-dim):
  - **lesion** — intensities inside the FLAIR `dseg` mask (binary; all 3 contrasts co-registered → the
    mask applies to each).
  - **overall** — the whole-image intensity histogram (no mask).
- One global p1–p99 normalisation over the whole image preserves where lesion intensities sit relative
  to the rest (bright FLAIR lesions vs dark T1w lesions).
- *Note:* with no brainmask, the `overall` histogram is ~70% background bin; its discriminative content
  is in the brain-tissue bins. This is the honest consequence of using only provided annotations.

## Data (all co-registered, 154×240×240 per subject)

- 30 subjects × {FLAIR, T1w, T2w}; lesion `dseg` present for all 30.
- **Real reference:** all 3 contrasts (90 scans) — Coverage per (contrast × scanner) group.
- **Synthetic sources:** FLAIR **and** T1w (analysed separately → `outputs/flair/`, `outputs/t1w/`).

## Pipeline

1. **Generate** (GPU, romane) — `generate_openms_volumes.py`: 4 methods
   (palette / synthseg_em / synthseg_noem / auglab_default) × {FLAIR, T1w} × 30 subj × 10 variants.
   The transform **seg is the lesion mask only** (`background:0, lesion:1`). **SynthSeg-EM** estimates
   its per-label GMM from the REAL image intensities, so it still fills the whole image sensibly from a
   coarse label map; **SynthSeg-noEM** (pure parametric) is expected to be poor on these sparse labels —
   that is the point. PALETTE's label-free core is unaffected. Reuses the dataset-agnostic per-method
   `configs/` from the on-harmony texture pillar.
2. **Extract** (CPU) — `extract_lesion_overall_openms.py`: real (3 contrasts) + synth (per source) →
   128-dim [lesion, overall] histograms. Validated on real: 90 scans, both histograms populate across
   all contrasts (co-registered lesion mask).
3. **Coverage + Vendi + PCA/UMAP** (CPU) — reuses the on-harmony `compute_coverage_metrics.py` /
   `plot_coverage_metrics.py` / `plot_manifold.py` **unchanged** (region-agnostic). Orchestrated by
   `run_coverage_openms.sh`.

## Honest expectations

- Same comparative claim as on-harmony (Coverage per real regime; PALETTE vs SynthSeg-EM vs
  auglab); **supporting/correlational**, not causal (causal = the Level-3 training ablation). With only
  2 coarse histograms, read it as (a) a check that the on-harmony finding is not parcellation-specific,
  and (b) the qualitative sparse-annotation point.
- **SynthSeg-noEM is expected to be poor here by construction** (sparse labels, no real intensities);
  **SynthSeg-EM is the fair SynthSeg baseline** (it uses the real intensities). Neither is a tuning
  artefact — it is the label-dependence of the method showing through.

Grounding is inherited from the on-harmony `GROUNDING_AUDIT.md` (metrics identical); the only
open-ms-specific choice is the 2-histogram [lesion, overall] feature, justified above.
