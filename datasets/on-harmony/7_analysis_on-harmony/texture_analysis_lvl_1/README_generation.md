# texture_analysis_lvl_1 — augmented volume generation

Generated source-aligned augmented copies of the 84 on-harmony T1w volumes using the
**exact AugLab GPU training operators** (`AugTransformsGPU`, the pipeline driven by
`nnUNetTrainerDAExt` via `AUGLAB_PARAMS_GPU_JSON`), with **all spatial/geometric
augmentation OFF**. Generation only — no metrics computed.

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
