# Prompt — Run the Level-1 texture-metric analysis (census + NMI, blur + no-blur)

> Paste below the divider to the Claude on the 4-GPU `set_slot` machine.
> Run **after** both volume sets exist: `data/generated/` (with-blur) and
> `data/generated_noblur/` (ablation). Scripts arrive via GitHub — just `git pull`.

---

## What this computes

Level-1 texture / structure preservation: for each generated volume vs its source T1w, per
anatomical ROI. The claim is **categorical**: image-driven augmentations (PALETTE, auglab_default)
preserve texture; label-generative SynthSeg destroys it (sits on the floor).

**Metrics** (see `LITERATURE_REVIEW.md` — do not change definitions):
- **census_r1 / census_r2 (PRIMARY texture)** — `|corr( rank(source), rank(synth) )|` on the
  eroded ROI, rank = census/rank transform (Zabih & Woodfill 1994; LBP ordinal family). Contrast-
  invariant (monotone) + inversion-invariant (`|·|`). **Floor = 0.** Radii r∈{1,2} reported for
  robustness; r=1 is the stringent headline.
- **NMI (secondary, content)** — Studholme normalized MI.
- Inline controls `gamma`, `histeq` = pure monotone remaps → census ≈ 1.0 (anchors the top).

No noise estimation, no η, no PyWavelets — census needs none of that. (`scikit-image` is only used
for the histeq control.)

## Run

1. `git pull`.
2. Self-test (must PASS — census_r1: identity/gamma/inverted→~1, noise→~0):
   ```
   .venv/bin/python .../texture_analysis_lvl_1/scripts/compute_texture_metrics.py --sanity --device cuda
   ```
3. Full analysis (both sets + controls → aggregate → plot):
   ```
   bash .../texture_analysis_lvl_1/scripts/run_texture_lvl1.sh
   ```
   It computes the **blur** set (`data/generated/`), the **noblur** set (`data/generated_noblur/`),
   and the `ref` controls, each sharded across `set_slot 0..3`, then aggregates + plots the
   combined result (blur vs no-blur side by side). Minutes of compute. Logs in `/tmp/texlvl1/`.

## Expected result (correctness check — verify, don't force)

`outputs/tables/summary.md`, **census_r1** column:
- **synthseg_em / synthseg_noem ≈ 0** (no-texture floor — texture destroyed), in **both** sets,
- **palette** and **auglab_default** far above the floor (image-driven, texture preserved);
  in the **noblur** set both rise (blur removed), PALETTE ≈ auglab (Voronoi costs little),
- **gamma / histeq ≈ 1.0** (set=ref) — pure monotone remaps, metric sanity.
- `stats.md`: PALETTE ≫ SynthSeg, rank-biserial ≈ +1, tiny p, in **both** sets.

Absolutes are **moderate and read relatively** (census r=1 is locally stringent — even the
accepted auglab baseline isn't near 1). The result is the **categorical gap** (image-driven ≫
SynthSeg ≈ 0), robust across r and across blur/no-blur. If SynthSeg is **not** near 0, or PALETTE
is **not** clearly above it, flag it — do **not** tune anything to force it.

## Report back

- `outputs/tables/summary.md` and `stats.md` (paste — both sets),
- confirm the PNGs in `outputs/plots/` (headline_census_r1, violin/heatmap per set, radius_robustness),
- combined CSV row counts per set and any volumes skipped (shape mismatch ⇒ generation broke
  alignment — flag, don't work around),
- any code tweak you made.
