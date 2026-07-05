# Prompt — Re-run Level-1 metrics with the new NGF metric

> Paste below the divider to the Claude on the 4-GPU `set_slot` machine.
> **No regeneration needed** — the 3360 volumes under `data/generated/` are unchanged.
> Just `git pull` to get the updated scripts, then recompute.

---

## Why we're re-running

The first pass used whole-ROI |LNCC| as the texture metric. It gave the right *direction*
(PALETTE > SynthSeg on all 84 subjects, p≈1.7e-15) but **compressed** the effect (PALETTE
0.48 vs SynthSeg 0.39 vs gamma 0.98), because a 9³ window straddles PALETTE's own Voronoi
sub-region boundaries and charges it for the coarse contrast restructuring that is its
*intended* behavior.

We've added a better, more faithful, verbatim-citable primary metric:

**NGF — canonical Normalized Gradient Fields** (Haber & Modersitzki 2006): per-voxel
`⟨n_η(∇src), n_η(∇syn)⟩²` with `n_η(I)=∇I/√(|∇I|²+η²)`, averaged over the eroded ROI. Measures
whether edges/texture point the same way (contrast- and inversion-invariant, squared handles
α<0). η (edge param) is set to the **source noise level as the paper prescribes**: η = √1.5 ×
Donoho–Johnstone wavelet-MAD estimate (`skimage.restoration.estimate_sigma`, needs **PyWavelets**).
The script auto-computes NGF across an **η grid ×{0.5,1,2}** (`NGF_ETA_MULTS`) for robustness →
columns `ngf_e0p5` / `ngf` (headline) / `ngf_e2p0`; random orientations give the **1/3 floor**.
Validated on synthetic phantoms (identity→~0.96, gamma→~0.95, PALETTE-like→~0.89, noise→~0.33;
`--sanity` PASSES).

NMI stays as a secondary content-preservation number; |LNCC| is kept only as a robustness
appendix. **Do not change any metric definitions** — they're pre-registered and validated.

## What to run

0. **Install PyWavelets** into the venv (needed by `estimate_sigma`): `.venv/bin/pip install PyWavelets`.
1. `git pull` (gets the updated `scripts/`).
2. Self-test (must PASS — note the new NGF rows):
   ```
   .venv/bin/python .../texture_analysis_lvl_1/scripts/compute_texture_metrics.py --sanity --device cuda
   ```
3. Full re-run (recompute → aggregate → plot; volumes already exist):
   ```
   bash .../texture_analysis_lvl_1/scripts/run_texture_lvl1.sh
   ```
   It re-shards compute across `set_slot 0..3`, overwriting `outputs/data/metrics_rank*.csv`
   with the NGF column added, then rebuilds tables + plots. Minutes of compute.

## Expected result (correctness check — verify, don't force)

In `outputs/tables/summary.md`, the **NGF** column should show:
- **synthseg_em / synthseg_noem ≈ 0.33** (at the random/no-texture 1/3 floor — texture destroyed),
- **palette** clearly above the floor, up near the image-driven controls (its blur is mild, so it
  is *not* dragged down; read it relative to the floor, not as an absolute — canonical NGF even
  scores identity <1),
- **gamma / histeq / auglab_default** highest (image-driven controls preserve texture).
- `stats.md`: PALETTE ≫ SynthSeg on NGF, large positive effect, tiny p.

If SynthSeg's NGF is **not** near the 1/3 floor, or PALETTE is **not** clearly above it, flag it —
don't tune η/erosion/anything to force it (that would be p-hacking; the metric is fixed).

The **η-robustness grid is automatic** now — `summary.md` has `ngf_e0p5`/`ngf`/`ngf_e2p0`
columns and `plots/eta_robustness.png` shows the direction holds across η. Nothing manual to do.

## Report back

- `outputs/tables/summary.md` and `stats.md` (paste them — NGF η-grid + NMI + LNCC),
- confirm the PNGs in `outputs/plots/` (violin_/heatmap_ × ngf/nmi/lncc, plus `eta_robustness.png`),
- combined CSV row count and any volumes skipped,
- any code tweak you had to make.
