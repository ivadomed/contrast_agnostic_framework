# Prompt — Generate a NO-BLUR ablation set (all 4 methods)

> Paste below the divider to the Claude on the 4-GPU `set_slot` machine.
> This is a **second, ablation** set — it does NOT replace the main `data/generated/` set.

---

## What & why

Generate a **no-blur version of all four methods** (palette, synthseg_em, synthseg_noem,
auglab_default), identical to the main generation **except blur and resolution degradation are
disabled**. This is a **mechanism-isolation ablation**: it separates the *contrast
transformation's* texture preservation from the blur/low-res augmentation (which attenuates
texture for any method). It will be reported **alongside** the main (training-config, with-blur)
set and **clearly labeled as an ablation** — the deployed method still uses mild blur, so the
with-blur numbers remain the headline. Do it **symmetrically for all methods** so the comparison
is fair (SynthSeg will stay near the texture floor regardless — it has no source texture to blur).

Everything else must be **identical** to the first generation (same 84 on-harmony T1w sources,
same 31-class labels, same isolated transforms, spatial augs OFF, N=10 stochastic variants,
outputs voxel-aligned to source, same `KEY` naming). Reuse the exact harness/configs you built
for `data/generated/`; change **only** the blur/resolution settings below.

## Output layout (parallel to the main set)

```
datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/data/generated_noblur/
  <method>/<KEY>/<KEY>_run-00.nii.gz ... _run-09.nii.gz
```
`KEY` = source filename minus `_0000.nii.gz` (includes `_T1w`). 84 × 4 × 10 = 3360 files.

## Exactly what to disable, per method (change ONLY these)

| method | disable blur/res by |
|---|---|
| **palette** | In `src/synthesis/v26_6_synthesis.py`, set `BLUR_SIGMAS = [0.0]` for this generation (v26_6_2 imports it, so this covers both). v26_6 has **no** resolution/zoom step, so blur is the only degradation. **Revert the edit after generating** (or override via a copy/env) — do not leave the training synthesis permanently changed. |
| **synthseg_em** | In the SynthSeg config block: `randomise_res=False`, and minimize acquisition blur (`blur_range=1.0`; `data_res=atlas_res=1.0`, `thickness=None`). Keep GMM fill, labels, EM settings identical. |
| **synthseg_noem** | Same as synthseg_em (SynthSeg block: `randomise_res=False`, `blur_range=1.0`, `data_res=atlas_res=1.0`). Keep the anatomical-label fill identical. |
| **auglab_default** | In the config, set `GaussianBlurTransform` **probability=0** and `SimulateLowResTransform` **probability=0**. Leave all other intensity augs unchanged. |

Do **not** change anything else (K-means/Voronoi parcellation, signed-α, GMM priors, intensity
augs, spatial-off, normalization). The only difference from `data/generated/` is no blur / no
resolution degradation.

## How to run

Same as the main generation: dispatch across `set_slot 0..3` (shard the 84 sources by rank),
`.venv/bin/python`, harness under this analysis dir's `scripts/`. Light work (~3360 fast
transform calls, no U-Net).

## Verify, then report

- `find .../data/generated_noblur -name '*.nii.gz' | wc -l` == **3360** (or report shortfalls).
- Spot-check 2–3 per method: correct shape+affine (aligned to source); palette/auglab visibly
  **sharper** than their with-blur counterparts; synthseg still GMM-filled label maps.
- Confirm you **reverted** the `BLUR_SIGMAS` edit (palette) so training code is unchanged.
- Report: counts per method, the config/code deltas you made, any deviations. **No metrics** —
  generation only.
