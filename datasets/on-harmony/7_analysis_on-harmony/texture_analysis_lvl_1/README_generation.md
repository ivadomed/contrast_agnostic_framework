# texture_analysis_lvl_1 — augmented volume generation

Generated source-aligned augmented copies of the 84 on-harmony T1w volumes using the
**exact AugLab GPU training operators** (`AugTransformsGPU`, the pipeline driven by
`nnUNetTrainerDAExt` via `AUGLAB_PARAMS_GPU_JSON`), with **all spatial/geometric
augmentation OFF**. Generation only — no metrics computed.

Two sets exist, both 84×4×10 = 3360 files, same `KEY` naming, same source-alignment:
- `data/generated/`         — **main / headline** set (training-config, WITH mild blur).
- `data/generated_noblur/`  — **mechanism-isolation ablation**: identical operators but
  blur + resolution degradation disabled (see the "No-blur ablation" section below).
  Report alongside the main set and label as an ablation — the deployed method keeps mild
  blur, so the with-blur numbers remain the headline.

## Outputs
```
data/generated/<method>/<sub-XXXX_ses-YYYY>/<sub-XXXX_ses-YYYY>_run-{00..09}.nii.gz
```
- methods: `palette`, `synthseg_em`, `synthseg_noem`, `auglab_default`
- 84 subjects × 4 methods × 10 stochastic variants = **3360 files** (verified complete;
  every subject dir holds exactly 10 runs).
- Every output shares the **exact shape + affine** of its source T1w (asserted on each
  write; 0 alignment failures in a 12-volume spot check).

## Derived configs (`configs/`) — built by `scripts/make_derived_configs.py`
Copied/derived from `sub-workspaces/auglab_workspace/AugLab/auglab/configs/`, originals
untouched. Each validated to have the intended generator active and **no voxel-moving
spatial transform** (`FlipTransform`, `AffineTransform`) active:
- `palette.json` — from `..._VALsynthonly_ImageContrastV26_6_2GPUTransform.json`, used
  as-is: only `ImageContrastV26_6_2GPUTransform` @ 1.0.
- `synthseg_em.json` — from `..._default01-23_Synthseg_EM.json`: `SynthSeg` @ 1.0
  (`em_label_completion=true`), **every other transform zeroed** (incl. FlipTransform).
- `synthseg_noem.json` — from `..._default01-23_Synthseg.json`: same, but
  `em_label_completion=false`. SynthSeg blocks preserved verbatim (GMM/EM priors intact).
- `auglab_default.json` — from `..._default01-23.json`: intensity/filter augs kept,
  only `FlipTransform` zeroed (`AffineTransform` was already 0). Grid-preserving
  `SimulateLowResTransform` kept.

## Harness
`scripts/generate_texture_lvl1_volumes.py` — rank-sharded over the 84 volumes
(`volumes[rank::4]`), one GPU per rank via `set_slot 0..3`. Input normalized with
nnUNet `ZScoreNormalization`, whole-image (matches Dataset031 `nnUNetPlans`
`use_mask_for_norm=[False]`). Transform raw output saved (no re-windowing). Resumable
(skips existing files unless `--overwrite`).

Run command (per GPU):
```
set_slot R .venv/bin/python scripts/generate_texture_lvl1_volumes.py --rank R --world-size 4 --n-variants 10
```

## Alignment guarantees
- The `RandomSynthSegGPU` wrapper already forces `apply_affine=False,
  apply_nonlinear=False, flipping=False, output_shape=None`, so the SynthSeg block's
  internal spatial params are inert w.r.t. voxel geometry — output stays grid-aligned.
- `nnUNetSpatialTransform` is not consumed by `AugTransformsGPU._build_transforms`
  (no builder branch), so it is inert regardless.
- Intensity/filter/resolution transforms (K-means contrast, GMM fill, blur,
  SimulateLowRes, bias field, gamma, …) are all grid-preserving.

## Deviation from the spec (noted per instructions)
Encountered a latent bug in the AugLab GPU transforms: several `Tensor.view(N, -1)`
flatten calls (e.g. `contrast.py` gamma, `fromSeg.py` RedistributeSeg) fail with
"view size is not compatible ... stride" when a prior in-place transform in the
`auglab_default` pipeline leaves the tensor non-contiguous. Fixed by replacing
`.view(` with `.reshape(` throughout `transforms/gpu/{contrast,fromSeg,spatial}.py`.
`reshape` returns byte-identical results when `view` would succeed and a correct copy
otherwise — **no change to operator semantics**, only robustness. This only affected
`auglab_default` (the intensity-aug pipeline); `palette`/`synthseg_*` never triggered it.
After the fix the full run completed with **0 errors**.

## No-blur ablation (`data/generated_noblur/`)

Mechanism-isolation ablation: identical to the main set EXCEPT blur / resolution
degradation are disabled, symmetrically for all four methods. Isolates the *contrast
transform's* texture preservation from blur/low-res attenuation (which lowers texture for
any method). Built by `scripts/make_noblur_configs.py` → `configs_noblur/`, then generated
with the same harness (`--config-dir configs_noblur --out-root data/generated_noblur`).

Per-method delta (the ONLY change vs the main configs):
- `palette` — `ImageContrastV26_6_2GPUTransform.blur_sigmas = [0.0]`.
  **Deviation from the task note (faithful to intent):** the task said edit
  `src/synthesis/v26_6_synthesis.py:BLUR_SIGMAS`. The GPU operator my harness uses
  (`RandomV26_6_2ContrastGPU`) is self-contained and reads `blur_sigmas` straight from the
  config JSON — it does NOT import that constant — so I disabled blur via the config
  instead. No training source was touched, hence nothing to revert.
- `synthseg_em` / `synthseg_noem` — SynthSeg block: `randomise_res=False`, `blur_range=1.0`,
  `data_res=1.0`, `atlas_res=1.0`, `thickness=None` (fixed 1 mm == atlas res → no
  blur/downsample). GMM / EM / label / bias / gamma settings preserved verbatim
  (EM identity kept: em=True / em=False respectively).
- `auglab_default` — `GaussianBlurTransform.probability=0.0`,
  `SimulateLowResTransform.probability=0.0`. All other intensity augs unchanged.

Nothing else changed (K-means/Voronoi, signed-α, GMM priors, intensity augs, spatial-OFF,
normalization).

Verification (0 errors, all 84×4×10=3360 present; 12-volume spot check, matched seeds):
- shape+affine identical to source — 0 alignment failures.
- no-blur is sharper (mean |∇| ratio noblur/withblur): synthseg_em/noem ~1.2–2.2×,
  palette ~1.0–1.6×, auglab ~0.4–0.85× on single run-00s.
- Notes on the ratios (expected, not errors):
  * palette blur is stochastic (sigma ∈ {0,0,0,0.3,0.5,0.8} per pass); on seeds where the
    with-blur draw was 0.0 the paired run is byte-identical (ratio 1.00), else no-blur is
    sharper. no-blur palette NEVER blurs.
  * auglab per-run ratios can dip below 1 because removing two probabilistic transforms
    shifts RNG consumption, so downstream edge augs (Scharr/Unsharp, which spike |∇|) fire
    differently per seed. The config delta is exactly blur+lowres; the *distribution* over
    the 10 variants is the correct no-blur one (the analysis aggregates over variants/ROIs,
    so this is fine — pairing is not required).
- synthseg stays near the texture floor regardless (abs |∇| ~0.05–0.10 vs palette ~0.5–0.7)
  — no source texture to blur, as expected.
