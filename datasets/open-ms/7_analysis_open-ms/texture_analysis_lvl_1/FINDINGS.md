# open-ms Pillar-1 texture analysis — findings & two resolved anomalies

Date: 2026-07-07. Metrics (all census = |corr(rank_transform(source), rank_transform(gen))|,
contrast/inversion-invariant, floor 0):
- `census_r1` (whole-ROI, r=1) — PRIMARY for texture-vs-texture-blind (image-driven ≫ SynthSeg).
- `census_r2` — radius-2 robustness.
- `census_local8` — same |corr| computed WITHIN 8³ blocks and averaged (cancellation-robust); use
  for palette-vs-auglab (Anomaly 2).
NMI was dropped: it is spatially blind (1-D intensity histogram, ignores voxel arrangement) so it
is not a texture metric — it stays only in the shared on-harmony pipeline.
ROIs: `lesion` (FLAIR dseg) and `foreground` (intensity>10% p99; NOT the computed brainmask).
n = 30 subjects × 10 variants, FLAIR and T1w sources. Focus: **noblur** set (blur removed so the
metric reflects the per-region FILL, not resolution simulation).

## Headline (final, noblur, spatial-off, blur-off), census_r1 per ROI (FLAIR)

| method                | foreground | lesion | reading |
|-----------------------|-----------:|-------:|---------|
| auglab_default        | 0.593      | 0.482  | conventional DA (mostly monotone) — see Anomaly 2 |
| palette (v26_6_2)     | 0.281      | 0.339  | image-driven contrast remap |
| synthseg_em           | 0.094      | 0.149  | near floor; slight lift from EM (see below) |
| v26_6_2_noisefill_v2  | 0.063      | 0.052  | **floor** (palette partition, noise fill) |
| synthseg_noem         | 0.006      | 0.069  | **floor** |
| gamma / histeq (ref)  | ~0.96      | ~0.96  | monotone ceiling (positive control) |

The texture-preservation vs texture-blind axis is clean and as designed:
**palette ≫ synthseg_em > synthseg_noem ≈ noisefill_v2 (floor)**, and noisefill_v2 (palette's
partition with the real-intensity fill swapped for noise) collapses to the floor — confirming the
fill, not the K-means/Voronoi partition geometry, carries the texture.

## Anomaly 1 — SynthSeg scored ABOVE the noise floor (now fixed)

**Symptom:** synthseg_em/noem census (0.12–0.17) sat well above the true i.i.d.-noise floor set by
v26_6_2_noisefill_v2 (~0.05), even though SynthSeg is mathematically texture-blind (per-region
Gaussian fill).

**Root cause (verified in source):** SynthSeg's generator always runs a resolution-simulation blur
(`generator.py:314-315` → `_simulate_resolution`). Its sigma is **hard-coded to 0.5 voxels when
`data_res == atlas_res`** (`functional.py:551`), so our "noblur" knobs (`data_res=1.0`,
`blur_range=1.0`) did NOT disable it — they only removed the random jitter on top of a σ=0.5
baseline. The mandatory blur smooths the painted region boundaries into graded transitions that sit
at the *exact same voxels* as the real image's tissue boundaries (SynthSeg paints onto the same
lesion/foreground mask, spatial deform disabled), manufacturing a boundary-driven rank correlation.

**Evidence (diagnostic, blur-on state):**
- Local smoothness (mean |x − local-neighbour mean|, z-scored volumes): synthseg 0.04–0.09 vs
  palette/noisefill/auglab 0.55–0.66 → SynthSeg output is ~10× smoother (the blur).
- Interior (eroded ×3) vs boundary-shell census, foreground: synthseg_em interior **0.077** vs
  shell **0.147** (+0.069 at the boundary) — its excess is boundary-driven; its interior already
  equals the noisefill floor (0.078). palette shows the opposite (interior 0.425 > shell 0.314):
  its signal is real interior texture, not boundary.

**Fix:** set `"apply_resolution": false` in the SynthSeg config block (threads through
`RandomSynthSegGPU` → `SynthSegGenerator(apply_resolution=False)`, skipping the blur). After the fix
synthseg_noem drops to **0.006** (foreground) — true floor — and matches noisefill_v2.

**Residual, and it is real (not an artifact):** synthseg_em stays slightly above floor (0.09 fg,
0.15 lesion) because `em_label_completion` fits an EM/GMM **on the real image's intensities** within
each label and hard-assigns voxels to real-intensity sub-clusters (`functional.py:722-816`) — i.e.
its sub-region *partition* is real-image-informed (analogous to PALETTE's k-means), even though the
painted values are random. So synthseg_em is genuinely, mildly *less* texture-blind than
synthseg_noem. Defensible finding, worth stating.

## Anomaly 2 — auglab_default scored ABOVE palette (explained; two confounds)

**Symptom:** whole-ROI census ranked auglab_default (0.54–0.59) above palette (0.28–0.34), opposite
to the expectation that the image-driven contrast method preserves the most texture.

**Confound A — cross-region sign cancellation (metric artifact, ~2/3 of the gap).** PALETTE applies
a **signed** per-region affine remap `μ_c + α_c·(x−mean_c)`, α_c ∈ ±[0.5,2.0] (`fromSeg.py:444-446,
472-474`). Within a region the local rank field is preserved (α>0) or inverted (α<0); the whole-ROI
`|corr|` sums across regions *before* `|·|`, so +corr and −corr regions cancel. Block-wise local
census (cancellation-robust) recovers palette:

| method | ROI | global | local8 | \|signed\|8 |
|--------|-----|-------:|-------:|-----------:|
| palette | fg  | 0.337 | **0.537** | 0.343 |
| auglab  | fg  | 0.561 | 0.611 | 0.564 |

For palette, local8 (0.537) ≫ |signed|8 (0.343) ≈ global (0.337) — the sign-flip fingerprint. For
auglab, local8 ≈ |signed|8 ≈ global — no flipping (its transforms are global-monotone). The
foreground gap shrinks from 0.224 (global) to 0.074 (local): most of the apparent gap was this
artifact.

**Confound B — auglab's census is bimodal / augmentation-strength (the residual).** auglab_default
applies a random subset of mild conventional transforms, each p≤0.4; **52% of its foreground samples
score census >0.8** (a monotone gamma/contrast, or near-identity, fired), IQR [0.07, 1.00]. palette
applies a strong remap *every* sample (frac>0.8 = 0.08, IQR [0.09, 0.37]). So auglab's higher *mean*
census reflects how often it *barely perturbs*, not better texture preservation — comparing means
conflates augmentation strength with texture fidelity.

**Consistent resolution (honest):** under the cancellation-robust `census_local8`, palette (~0.54)
and auglab (~0.58–0.61) are **~tied** (auglab marginally higher). There is **no honest metric that
ranks palette clearly above auglab on texture** in this config — because auglab_default here is
dominated by monotone, rank-preserving ops that genuinely preserve local texture. The census *mean*
is the right discriminator for the texture-vs-texture-blind axis (palette ≫ synthseg/noisefill,
decisive) but NOT for palette-vs-auglab, which differ in augmentation *strength/monotonicity* rather
than texture-fill fidelity. (An earlier draft cited NMI as ranking palette>auglab; that was
withdrawn — NMI is spatially blind and reflects intensity-mapping determinism, not texture, so it is
not a valid texture argument. NMI has been removed from the open-ms analysis.) Bottom line: report
`census_local8` for palette-vs-auglab and state they preserve local texture comparably; reserve the
headline for palette ≫ the texture-blind floor.

## Config/consistency fixes applied
- Spatial augmentation was ON for synthseg_em/noem/auglab_default but OFF for palette/noisefill in
  the shared on-harmony configs. Made open-ms-local copies with `p_rotation=p_scaling=0` for a
  controlled comparison (`data/configs_noblur_nospatial/`); shared on-harmony configs left untouched.
- `aggregate_texture_metrics.py`: summary now reports median alongside mean, with a caution that the
  table pools ROIs (read `per_roi_*.csv` for the clean per-ROI comparison); **auto-detects present
  metric columns** (`metrics_present`) so on-harmony (census_r1/r2/nmi) and open-ms (census_r1/r2/
  local8, no nmi) both work from the one shared script.
- `plot_texture_metrics.py`: `spresent()` renders any set label (open-ms uses flair/t1w); `per_vol`
  auto-detects metric columns; headline/violin/heatmap now emitted for each present census variant
  (census_r1 AND census_local8); radius plot guards missing census_r2.
- `census_local8` added to `compute_texture_metrics_openms.py` (`local_abscorr`, vectorised;
  validated vs brute-force to 2e-8 and by the --sanity self-test). NMI dropped from open-ms output
  (localized to this script; on-harmony unaffected).

## Diagnostic scripts (reproducible)
- `scripts/diagnose_synthseg.py` — H1 smoothness + H2 interior/shell census (Anomaly 1).
- `scripts/diagnose_local_census.py` — global vs block-wise local census + sign-cancellation
  fingerprint (Anomaly 2).
