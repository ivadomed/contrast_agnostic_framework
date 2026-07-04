# Prompt — Generate augmented on-harmony volumes (PALETTE + SynthSeg + controls)

> Paste everything below the line to the Claude instance on the 4×48 GB GPU machine.
> That machine uses `set_slot 0-3` (not Slurm). It has this same repo + `.venv`.

---

## Your task

Generate **augmented copies of the 84 on-harmony T1w volumes**, applying four augmentation
methods, **N = 10 stochastic variants each**, and save every output **voxel-aligned to its
source volume** (identical shape + affine). These volumes feed two later analyses (which you
do NOT run — you only generate the volumes):
1. a **texture-preservation analysis** (voxelwise source↔synth metrics), and
2. a **re-run of the contrast-manifold analysis** using all 31 anatomical classes.

**Why alignment matters:** both downstream analyses compare each synthetic voxel to the
*same* source voxel (and to per-class anatomical masks). If the output is cropped, flipped,
resampled, or spatially deformed relative to the source, the analysis is invalid. Therefore
**every spatial/geometric augmentation must be OFF** — apply intensity/contrast transforms
only, keeping the voxel grid fixed.

**Correctness anchor (most important):** the augmented volume must be produced by the
**exact augmentation operators used at training time** — the **AugLab GPU transforms** driven
by the training config JSONs. Do **not** use `scripts/generate_synthetic_guidance.py`
(guidance-map path; doesn't support v26_6_2) or
`scripts/utils/generate_synthseg_comparison.py` (BrainGenerator path). Those are different
operators; the texture claim must transfer to the *actual method*.

## Inputs

- Source images: `datasets/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/imagesTr/*_0000.nii.gz` (84 volumes)
- Anatomical labels (31 classes + bg): `.../Dataset031_OnHarmonyT1w31/labelsTr/<same_name>.nii.gz`
- Pairing is by filename: `sub-XXXX_ses-YYYY_T1w_0000.nii.gz` ↔ label `sub-XXXX_ses-YYYY_T1w.nii.gz`.
- AugLab configs dir: `sub-workspaces/auglab_workspace/AugLab/auglab/configs/`

## The four methods and the config to drive each

Use the AugLab GPU transform pipeline (`auglab/transforms/gpu/transforms.py` builds a
`ComposeTransforms` from a config's `GPU` block; the trainer that uses it is
`auglab/trainers/nnUNetTrainerDAExt.py`, env var `AUGLAB_PARAMS_GPU_JSON`). For each method,
drive it with a **synth-only, spatial-off** config:

| Method (output dir) | Base config | What to keep active |
|---|---|---|
| `palette` | `transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json` | already clean: ONLY `ImageContrastV26_6_2GPUTransform` @ 1.0 — use as-is |
| `synthseg_em` | `transform_params_gpu_default01-23_Synthseg_EM.json` | keep the `SynthSeg` block @ 1.0, **zero every other transform** (incl. `FlipTransform` @ 0.5 — spatial!) |
| `synthseg_noem` | `transform_params_gpu_default01-23_Synthseg.json` | same: `SynthSeg` @ 1.0, zero all others incl. `FlipTransform` |
| `auglab_default` | `transform_params_gpu_default01-23.json` | keep the intensity augs; **zero only the spatial ones** (`FlipTransform`, `AffineTransform`, any elastic/deformation). `SimulateLowResTransform` is grid-preserving → keep it. |

- For `synthseg_em` vs `synthseg_noem`: **preserve each source config's `SynthSeg` params
  block verbatim** (that block encodes the EM-clustering-vs-anatomical-labels difference and
  the GMM priors) — you are only zeroing the *other* transforms.
- Create these as new derived JSONs (e.g. under this analysis dir's `configs/`), so the
  originals are untouched and your exact settings are reproducible. Verify each derived config
  has **exactly one** generative transform active and **no spatial** transform active before
  running.

## Input normalization

Feed each volume to the transform normalized the **same way training does** (nnUNet
foreground Z-score, per `nnUNetTrainerDAExt` / the datamodule — inspect it to match).
`ImageContrastV26_6_2` internally re-normalizes in-mask, so exact input scale is not critical,
and both downstream analyses re-normalize anyway — but stay faithful to training so the
operators' internal K-means/GMM behave identically. Save the transform's **raw output** (do
not re-window it).

## Output layout (create under this analysis dir)

Let `KEY` = the source filename with `_0000.nii.gz` removed — i.e. it **includes the `_T1w`
suffix**, e.g. `sub-03286_ses-NOT2ING001_T1w`. Use `KEY` verbatim for both the folder and
the file stem (the metrics script keys volumes by exactly this):

```
datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/data/generated/
  <method>/<KEY>/<KEY>_run-00.nii.gz ... <KEY>_run-09.nii.gz
```
- `<method>` ∈ {palette, synthseg_em, synthseg_noem, auglab_default}
- Each `run-NN` is an **independent stochastic draw** (different RNG seed per variant).
- Every output NIfTI: **same shape and same affine as the source** T1w. (Sanity-assert this
  on write — `np.allclose(out.affine, src.affine)` and equal shape.)
- Total: 84 volumes × 4 methods × 10 variants = **3360 files**.

## How to run (this machine = `set_slot`, NOT Slurm)

- **Never run the generation on the login node.** Dispatch through `set_slot 0`–`set_slot 3`
  (one GPU each) — see the repo's `docs/HANDOVER.md` §1 for the exact `set_slot` pattern.
- Parallelize across the 4 GPUs by splitting the 84 volumes by rank
  (`volumes[rank::4]`, rank ∈ 0..3), each rank on its own `set_slot`. Run all four methods
  inside each rank (or loop methods outer, ranks inner — your call).
- Use the repo venv: `.venv/bin/python`. Keep a starter/harness script under this analysis
  dir's `scripts/` (version-controlled), not in `/tmp`.
- This is light work (~3360 fast GPU transform calls, no U-Net) — should finish quickly even
  at low cluster priority.

## Implementation approach (recommended)

Write one small harness `scripts/generate_texture_lvl1_volumes.py` that:
1. lists the 84 source volumes (rank-sharded);
2. builds the AugLab `ComposeTransforms` from a given derived config (reuse
   `auglab/transforms/gpu/transforms.py`);
3. for each volume: load T1w (+ label), normalize (as training), run the transform N times
   (fresh seed each), save each output aligned to source affine/shape;
4. asserts alignment on every write; logs a per-method count at the end.
Patterns for load→transform→save-aligned and the `set_slot` 4-GPU loop already exist in
`scripts/utils/generate_synthseg_comparison.py` and `scripts/generate_synthetic_guidance.py`
— copy the plumbing, swap in the AugLab transform. You have the full repo; wire the exact
APIs by reading the code. If something in my spec conflicts with what the code actually
supports, **prefer correctness of the stated intent** (exact training operator, spatial off,
source-aligned output) and note the deviation.

## Verify, then report back

- Confirm `find .../data/generated -name '*.nii.gz' | wc -l` == **3360** (or report which
  method/rank fell short and why).
- Spot-check 2–3 outputs per method: correct shape+affine; `palette`/`auglab_default` visibly
  retain anatomy (contrast changed), `synthseg_*` look like GMM-filled label maps.
- Report back: total counts per method, the derived config paths you created, the harness
  script path, and any deviations. **Do not compute any metrics** — generation only.
