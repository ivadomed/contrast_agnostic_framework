# 00_04_analysis — cross-dataset method-comparison analyses

Method-agnostic re-analyses of the per-case metric CSVs written by
`00_03_evaluate/evaluate.py` (`<exp>/fold{F}/<testmod>_metrics.csv`). These don't run
models — they mine existing predictions to characterise *where* and *why* one
augmentation method beats another.

## `texture_advantage.py`

Quantifies where **v26+auglab (image-driven)** beats **SynthSeg-EM (label-driven)** per
structure and per test-contrast: Δmean Dice, ΔP25 (tail), Δfailure-rate. Writes
`texture_advantage_per_label.csv` + `summary.md` into a dataset's
`8_results_*/02_metrics/<model>/<contrast>/exp_texture_advantage/`.

Run (examples that produced the results below):
```bash
PY=.venv/bin/python; A=datasets/00_commun_scripts/00_04_analysis/texture_advantage.py
$PY $A --metrics_base datasets/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w \
  --ours_regex auglabAug_v26_6_2 --contender_regex synthseg_EM --contender_name synthseg_EM \
  --out_dir datasets/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w/exp_texture_advantage
# (BraTS t1n and CHAOS t1in analogously — see each dataset's exp_texture_advantage/summary.md)
```

## Finding (2026-06-30): the v26 advantage is a *texture / spatial-structure* effect

**Mechanism (verified in `AugLab/auglab/transforms/`):**
- SynthSeg renders each label region as `mean + std·N(0,1)` — a flat mean + white noise.
  EM only subdivides the mosaic into more flat-noise pieces. It is **texture-blind** and
  produces **hard, gradient-free boundaries**.
- v26 applies `y = μ + α·(x − mean_region)` to the REAL image — an *affine* (texture-,
  gradient-, partial-volume-preserving) contrast remap.

**Consequence:** on plain mean-Dice over large, mean-separable structures (whole organs,
big brain ROIs) the two are near-tied (+1–4 Dice) — SynthSeg's best case, and the reason
prior comparisons looked "incremental". The v26 advantage is real and **large** exactly
where spatial intensity structure is the signal, and is verified robust per-fold:

| Where | Δ Dice (ours − ssEM) | Notes |
|---|---|---|
| BraTS **edema (SNFH)** | **+7 to +11**, all folds × t1c/t1n/t2w (n=280) | + boundary 3–6 mm sharper; uniform across edema texture (boundary/gradient effect) |
| on-harmony **thin ventricle horns** (InfLatVent) | **+11 to +14**, every fold | tiny CSF structures |
| on-harmony **Caudate on EPI** | **+9/+10/+13/+19**; **failure-rate −56%** | hardest cross-contrast |
| on-harmony **Hippocampus on GRE** | +8 to +10, every fold | |
| **Failure tail** (broadly) | synthseg fails (Dice<0.5) on **20–56% more cases**; ΔP25 up to +25 | reliability |
| **Boundary HD95** | ours sharper in 13/16 (BraTS) & 11/13 (CHAOS) cells | mixed on dense brain |

**Where SynthSeg wins** (mechanism-consistent): ultrasound (TRUSTED), BOLD/DWI ventricles,
AMOS-CT left-kidney — extreme modality shifts where the *real texture v26 preserves does
not transfer* to the target domain, and only the (arbitrary) GMM contrast helps.

**Takeaway for benchmarking:** stop headlining whole-organ cross-contrast mean-Dice (a
near-tie by construction). Lead with fine/thin structures, texture-/gradient-defined
pathology (edema), boundary accuracy, failure-rate, and the most exotic contrasts.
