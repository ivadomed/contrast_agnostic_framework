# Overnight autonomous work — 2026-07-30 night

## The headline result: NGF dose-response confirms the texture mechanism

**`paper/ngf_dose_response_openms_flair.png`** — the single most important output of
tonight. Left panel: OOD Dice across all 7 ladder rungs. Right panel: independent NGF
texture measurement (foreground-**interior**, erode×3 — the ROI that matches the
Killarney headline plot Paul pointed me to; my first pass used the wrong,
non-eroded ROI and got auglab_default > palette, the wrong ordering).

**The finding:** the 3 noise-fill partition rungs (kmeans → +label_remap → +voronoi)
sit FLAT at the texture floor (NGF 0.451–0.457, same as synthseg_em's 0.462) all
the way through rung 4 — their Dice gains (11.2 → 17.5 → 19.6) come from *something
other than texture* (more likely just richer contrast augmentation from finer
parcellation). Then the rung 4→5 fill swap (noise→real, **identical partition**)
jumps NGF to 0.809 — clear of even auglab_default's 0.768 — while Dice jumps
+7.70. Independent measurement and downstream Dice agree on exactly where the
texture effect turns on. That is about as clean a causal signature as this kind of
ablation can produce.

Also fixes what you flagged: this pass correctly reproduces `palette > auglab_default`
(0.809 vs 0.768) on the interior ROI, matching `FINDINGS.md`'s documented result and
the Killarney plot. Root cause of my earlier wrong number: Vulcan never had the
eroded/interior CSVs at all — only the non-eroded whole-foreground variant, which
gives the wrong ordering because auglab_default trivially preserves gradients at
tissue *boundaries* (it's mostly monotone) and erosion is specifically what strips
that boundary shell out.

## Ladder plots (Dice+HD95, dissociation)

`paper/ablation_ladder_openms_chaos.png` — same rung 3→4 fill-swap isolation, but
as the downstream Dice/HD95 curve for open-ms FLAIR (texture-defined) vs CHAOS T1in
(boundary-defined):

| dataset | OOD Dice: noise→real fill | Δ |
|---|---|---|
| open-ms FLAIR | 19.57 → 27.27 | **+7.70** |
| CHAOS T1in | 87.15 → 88.22 | **+1.07** |

Texture is worth ~7× more on the texture-defined task — the dissociation predicted
in advance, now with the NGF result independently confirming *why*.

## What's still running

**CHAOS T2spir ladder** (jobs 389063/389064, TamIA): training now, healthy (no
crash, real wandb/epoch progress as of last check), 200 epochs — should complete
within a few hours of you reading this. I have a monitor armed for completion;
once it lands I still need to predict+evaluate it (same `06_01_evaluate_run.sh`
pattern used all session) before it has real Dice numbers — that part is NOT done
yet, I'll pick it up as soon as training finishes.

**BraTS T1n ladder** (jobs 389047-389054, chain of 8×24h, TamIA): training,
confirmed healthy (real AugLab config loaded, splits found, into torch.compile).
**This will not finish overnight** — 2500 epochs × 4 new rungs × 3 folds, packed
3-per-GPU. Realistic estimate is several days, not one night. The launcher
(`04_43_tamia_pack_t1n_ladder.sh`) persists RUN_IDs in
`/scratch/p/paulh/brats2024-glioma/_packruns/t1n_ladder_20260730_200711/RUN_IDS.env`
— re-running it with the same `PACK_DIR` extends the chain with zero re-recording
if 8 links isn't enough. I chose **T1n** over T2w on the reasoning that T1n's
toughest-competitor margin was consistently larger everywhere I computed this
session (e.g. vs auglab_default +0.85 vs T2w +0.49) — more headroom to see a ladder
if one exists. Unresolved caveat I flagged to you and haven't answered: I'm not
confident BraTS's 3 labels (NCR/SNFH/ET) cleanly split into texture- vs
boundary-defined the way MS-lesion-vs-organ does. This is a within-dataset test of
the same hypothesis, not a guaranteed second confirmation.

## Two incidents tonight, both my process error, both fixed

1. **AugLab configs weren't on Vulcan at all** — Killarney had 15
   `baseline_kmeans*`/`auglab_kmeans*` configs neither Vulcan nor TamIA had. This is
   what let me build the ladder plot in the first place (rungs 2-4 needed these).
   Synced to both.
2. **Both TamIA training launches (CHAOS, then the first NGF attempt) crashed
   immediately** because I launched them *before* finishing that sync to TamIA
   specifically (I'd only synced for Vulcan/BraTS's benefit at that point). Both
   resubmitted clean once the configs landed. CHAOS's *second* attempt then hit an
   unrelated one-off "No CUDA GPUs available" (transient — didn't recur on retry).
   No project-code bug in either case — worth knowing about only so it isn't
   mistaken for something structural.

## Everything created tonight (for the record)

- `datasets/open-ms/7_analysis_open-ms/histogram_coverage_lvl_1/scripts/generate_openms_volumes.py`
  — added `baseline_kmeans`, `baseline_kmeans_label_remap`, `v26_6_2_noisefill_v2`
  to the `--methods` choices list.
- `datasets/open-ms/7_analysis_open-ms/data/configs_noblur_nospatial/{baseline_kmeans,baseline_kmeans_label_remap}.json`
  + the on-harmony-style pair — derived from the AugLab training JSONs'
  `skip_sub_parc_prob`/`label_remap_prob` knobs (rung2: sub_parc=1.0, remap=0.0;
  rung3: sub_parc=1.0, remap=0.5), verified same transform class
  (`ImageContrastV26_6_2NoiseFillGPUTransform`) as the already-adopted rung.
- `datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts/run_ngf_eroded_openms.sh`
  (superseded by the full-pipeline version below, kept for reference).
- `scripts/cluster/tamia_env_chaos.sh` — own-training scratch override for CHAOS on
  TamIA (mirrors `tamia_env.sh`'s pattern for brats2024-glioma; distinct from the
  `_chaoscross` files, which point the other direction).
- `datasets/chaos/5_scripts_chaos/04_train/04_5{5,6,7}...sh` — pre-existing, never
  run; just launched them.
- `datasets/brats2024-glioma/5_scripts_brats2024-glioma/04_train/04_40/41/42...sh`
  (3 new ladder-rung wrappers) + `04_43_tamia_pack_t1n_ladder.sh` (pack launcher,
  modeled exactly on the existing `04_33` pattern).

## What to check first, in order

1. `paper/ngf_dose_response_openms_flair.png` — the headline result, already done.
2. CHAOS t2spir training (389063/64) — likely done; if so I still owe you
   predict+evaluate before there's a real Dice number for that rung.
3. BraTS (389047-54) — check `sacct` for chain progress; expect it mid-flight,
   not close to done.
