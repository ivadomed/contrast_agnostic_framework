# Prompt — Run the Level-1 texture-metric analysis

> Paste everything below the line to the Claude on the 4-GPU `set_slot` machine.
> **Run this only AFTER the volume generation is done** (the 3360 files under
> `data/generated/`). Scripts arrive via GitHub.

---

## What this is

You are computing **Level-1 texture / structure-preservation metrics** that compare each
generated volume to its source T1w, per anatomical ROI. The goal is to show, quantitatively
and with citable metrics, that **PALETTE preserves the source's texture/structure while
SynthSeg destroys it** (SynthSeg fills each region with spatially-independent GMM noise).

Everything is already written and validated on another machine:
- Metric math **self-test passed** (identity→NMI 2.0/|LNCC| 1.0; noise→1.0/0.03; gamma→1.77/0.99).
- Aggregation + plotting **verified end-to-end** on synthetic data.
The only thing not yet exercised is the real-data I/O path (loading the volumes) — that's
where a small tweak *might* be needed. Your job: run it, and fix any path/format mismatch,
staying faithful to the intent below.

## Metrics (do not change the definitions — they're chosen for citability)

- **NMI** — Studholme normalized mutual information `(H(X)+H(Y))/H(X,Y)` ∈ [1,2]. Contrast/
  inversion-invariant content-preservation measure (Maes 1997 / Studholme 1999).
- **|LNCC|** — |local normalized cross-correlation|, box window (Avants 2008 / ANTs);
  invariant to local linear intensity change, `|·|` for PALETTE's α<0 inversion.
- Computed on **eroded** 31-class ROI masks. Two inline positive controls (`gamma`, `histeq`)
  are synthesised from the source inside the script — no extra volumes needed.

## Files (under `datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/scripts/`)

- `compute_texture_metrics.py` — per-volume, per-ROI metrics → long CSV. GPU (torch),
  supports `--rank/--world-size` sharding and `--sanity` self-test.
- `aggregate_texture_metrics.py` — combines shards, averages variants, per-method summary,
  per-ROI tables, paired Wilcoxon (PALETTE vs SynthSeg) + effect sizes.
- `plot_texture_metrics.py` — violin per method + per-ROI × method heatmaps.
- `run_texture_lvl1.sh` — the driver (sanity → 4 GPU shards via `set_slot 0..3` → aggregate → plot).

## How to run

1. **Self-test first** (must PASS before anything):
   ```
   .venv/bin/python .../scripts/compute_texture_metrics.py --sanity --device cuda
   ```
2. **Full pipeline**:
   ```
   bash .../scripts/run_texture_lvl1.sh
   ```
   It shards the ~3360×(31 ROI) work across the 4 GPUs with `set_slot 0..3`, then aggregates
   and plots. Logs in `/tmp/texlvl1/`. (Light work — minutes of compute; the driver already
   uses the `set_slot` pattern from `docs/HANDOVER.md` §1.)

## Where a tweak MIGHT be needed (and the intent to preserve)

The compute script keys each volume by `KEY` = source filename minus `_0000.nii.gz`
(**includes `_T1w`**, e.g. `sub-03286_ses-NOT2ING001_T1w`), and expects:
- source: `Dataset031_OnHarmonyT1w31/imagesTr/<KEY>_0000.nii.gz`
- label:  `Dataset031_OnHarmonyT1w31/labelsTr/<KEY>.nii.gz`
- synth:  `data/generated/<method>/<KEY>/<KEY>_run-NN.nii.gz`

If the generation step named folders/files differently, **fix the globbing in
`list_source_keys` / `build_tasks`** (or rename the outputs) so pairing works — don't change
the metric definitions. Other likely-small fixes: `--device` if cuda enumerates oddly under
`set_slot`; the `subject/session` split in the main loop if a filename doesn't match
`sub-*_ses-*_T1w`; `skimage`/`torch` import availability in the venv. The script skips (with a
warning) any volume whose shape ≠ source — if you see many of those, generation broke
alignment (spatial aug leaked in) and should be re-done, not worked around.

## Correctness check (this is the whole point — verify it)

In `outputs/tables/summary.md` you should see roughly:
- **synthseg_em / synthseg_noem: NMI ≈ 1.0–1.1 and |LNCC| ≈ 0** (texture destroyed),
- **palette, auglab_default, gamma, histeq: NMI ≳ 1.5 and |LNCC| ≳ 0.7** (texture preserved).
- `stats.md`: PALETTE ≫ SynthSeg, large positive effect, small p.

If SynthSeg does **not** collapse to ~0, something is wrong (e.g. synth not actually GMM-noise,
or misaligned) — flag it rather than reporting it as a result.

## Report back

- `outputs/tables/summary.md` and `outputs/tables/stats.md` (paste them),
- confirm the 4 heatmap/violin PNGs in `outputs/plots/`,
- row count of the combined CSV and any volumes skipped (with reason),
- **any code tweaks you made** (so they can be merged back).
