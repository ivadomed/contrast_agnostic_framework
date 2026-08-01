# KIDNEY-T2W: geometry-confound investigation — CONCLUDED

**Bottom line: there is no geometry confound to normalize away, and no pipeline
bug. The un-normalized numbers are correct.** The `srcsm` collapse is a real,
`srcsm`-specific fragility. OURS trailing `synthseg_EM` on the t1in branch is a
real result. My initial "acquisition-plane confound" hypothesis was **refuted**
by the full ablation — details below, including why an early 8-case pilot
misled me into believing it.

---

## Trigger

srcsm scored 41.0 Dice (t1in, 3-fold) vs its usual ~80-89 on every other
cross-eval set; OURS ~74 vs synthseg_EM's ~80. Both flagged as implausible.

## Ruled out (each tested, not assumed)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | FOV wiring bug | **No** | All 6 cross-eval datasets audited: correct anchor organ + correct own-GT id from `chaos_fov_margins.json`. |
| 2 | FOV cropping real kidney | **No** | Full uncropped-volume Dice on 8 worst cases matched FOV-restricted values (same 0.06-0.54 range). |
| 3 | CKD pathology | **No** | HC vs CKD near-identical — srcsm fold0 0.256 vs 0.264; OURS 0.784 vs 0.734; synthseg_EM 0.861 vs 0.837. |
| 4 | nnU-Net resamples the *wrong axis* | **No** | Source check (`get_do_separate_z`/`get_lowres_axis`): anisotropy detected per-case from actual spacing. |
| 5 | Interpolation *quality* | **No** | Job 391580, 1.5mm isotropic pre-inference: 0.339 → **0.332**. No gain. |
| 6 | Too little context / needs padding | **No** | Job 391677, pad sparse axis to 48: predictions **byte-identical** to no-pad (nnU-Net already pads internally). No-op. |
| 7 | **Acquisition-plane geometry confound** | **No — refuted** | Job 391686, see below. |
| 8 | Wrong trainer/run-id (loading wrong checkpoint) | **No** | All 16 shared RUN_IDs cross-checked against cirrmri-liver + msd-spleen: every `TRAINER` string identical. |

## The refuted hypothesis (#7) and why the pilot misled me

*Reasoning was:* nnU-Net resamples to chaos's plans spacing `(5.5, 1.699, 1.699)`
(internal `z,y,x`), whose coarse axis is NIfTI array **axis2**. chaos
`(256,256,35)` @ `(1.89,1.89,5.5)` has its coarse axis there; kidney-t2w
`(240,14,240)` @ `(1.458,5.5,1.458)` has it at **axis1**. So nnU-Net
*downsamples* kidney-t2w's rich S-I axis 3.8× and *upsamples* its sparse P-A
axis 3.2× — apparently catastrophic.

*Two candidate normalizations were tested* (both moving the coarse axis to
axis2): **`swap`** = pure axis permutation, lossless, in-plane view becomes
coronal; **`axial`** = resample into a chaos-like axial grid (in-plane L-R×P-A
@1.7mm, through-plane S-I @5.5mm), matching chaos's *view* but interpolating P-A
from only 14 real samples.

### Job 391686 — 100 cases × 4 methods × 3 variants, fold0

Mean Dice ×100 (median in parens):

| method | orig | swap | axial |
|---|---|---|---|
| baseline | 0.1 | 0.0 | 0.1 |
| synthseg_EM | **84.3** (86.4) | 20.5 (11.7) | **84.4** (86.2) |
| srcsm | 25.9 (25.4) | **39.7** (42.7) | 26.6 (24.1) |
| OURS_val100 | **75.8** (78.8) | 23.2 (14.2) | **77.1** (79.8) |

Cases with Dice < 0.10 (catastrophic):

| method | orig | swap | axial |
|---|---|---|---|
| baseline | 100 | 100 | 100 |
| synthseg_EM | **0** | 49 | **0** |
| srcsm | 19 | 9 | 20 |
| OURS_val100 | **0** | 42 | **0** |

**Conclusions, against the criterion written down before running it:**

1. **No information is being destroyed.** synthseg_EM reaches **84.3** and OURS
   **75.8** on the completely un-normalized `orig` data, with **zero**
   catastrophic failures. If the resample-target mismatch were destroying the
   kidney signal, no method could score 84%. Hypothesis #7 is dead.
2. **`axial` ≈ `orig`** (84.4/77.1/26.6 vs 84.3/75.8/25.9). Matching chaos's
   acquisition *view* changes essentially nothing. Confirms geometry is not the
   limiting factor.
3. **`swap` is actively harmful and must NOT be adopted** — it destroys
   synthseg_EM (84.3→20.5, 0→49 failures) and OURS (75.8→23.2, 0→42). It is
   *only* beneficial for srcsm.
4. **My 8-case pilot was a badly designed experiment.** It tested `swap` on
   **srcsm alone** — the single method of four that benefits — and on 8 cases.
   It reported 0.339→0.414 and looked like a fix. Testing all methods reversed
   the conclusion completely. *Lesson: never validate a shared preprocessing
   change on one method, and never on 8 cases.*

## What the anomaly actually is

**srcsm is uniquely fragile here, and it is not case difficulty:**

- On the exact 19 cases where srcsm fails (<0.10), **synthseg_EM averages 79.9
  and OURS 64.3** — those cases are plainly segmentable.
- srcsm's failures don't track HC/CKD (13 HC vs 6 CKD) or slice count (all 19
  have 13 slices; non-failures span 11-14).
- srcsm's whole distribution is depressed, not bimodal — deciles
  `0, 5, 10, 16, 20, 25, 32, 35, 40, 47, 56`. It **never exceeds ~0.56** on any
  of 100 cases, while synthseg_EM's median is 86. This is systematic
  mis-segmentation, not occasional failure.
- srcsm is **the only method helped by `swap`** (25.9→39.7, 19→9 failures), so
  it *is* the one method with real sensitivity to which axis carries the
  anisotropy.
- srcsm's checkpoint is healthy: 88.8 in-domain chaos_t1in (best of all
  methods), 79.9 msd-spleen, 80.5 cirrmri-liver t1, and **84.9 on amos_mri**
  (which includes kidney labels) — i.e. srcsm segments kidneys fine on *axial*
  MRI and fails on *coronal* MRI.

**Reading:** SRCSM (semantic random convolution) appears to have learned
features tied to the acquisition geometry / spatial-frequency layout of axial
data, making it uniquely brittle to out-of-plane acquisition. That is a genuine
property of the method, and arguably a finding in its own right — the
domain-randomization methods (synthseg_EM, OURS) are robust to the plane change
that breaks srcsm. It cannot be "normalized away" for the suite, because the
only transform that helps srcsm wrecks every other method.

**baseline ≈ 0 is expected, not a bug** — no augmentation, trained on T1in,
tested on T2w. It also scores 6.3 on chaos's own t2spir. That contrast gap is
precisely what this suite is built to measure.

**OURS vs synthseg_EM:** on this cohort synthseg_EM genuinely leads the t1in
branch (84.3 vs 75.8 fold0; 80.5 vs 74.1 across 3 folds). On the t2spir branch
they are near-tied (82.9 vs 82.1). This is consistent with the known
[incremental-margin picture](../../) vs SynthSeg and is not explained by
geometry (#7 refuted) or by any config error (#8 checked). Reported as-is.

## Practical outcome

**No change to the pipeline or the published numbers.** kidney-t2w's committed
results stand as correct. The dataset is a legitimate cross-contrast test set;
its one notable feature is that it is the first test set whose acquisition plane
differs from chaos's, which surfaced srcsm's geometry fragility.

## Reproduction

```bash
# TamIA, whole-node H100, ~2 min
sbatch datasets/kidney-t2w/5_scripts_kidney-t2w/05_predict/05_30_geometry_normalization_ablation.sh
```
Per-case output: `geom_norm_per_case.csv` in the job's work dir. Superseded
pilots (kept for provenance, `/scratch/p/paulh/kidney-t2w/_packruns/`): 391580
isotropic, 391593 swap, 391677 swap+pad.
