# healthy-spine-TUM — externally-computed results

> **Re-imported 2026-10-02** from the collaborators' second drop, which adds HD95.
> The importer is now `5_scripts_healthy-spine-tum/06_evaluate/06_00_import_tum_tsv.py`
> (replaces the one-off `_import_tum_tsv_20260805.py`, kept for provenance only).
> Two things changed besides HD95, both verified row-by-row:
> - Folder names: Ours is `ImageContrastV26-05+Paper` and Auglab is `newd` in the
>   new drop (each reproduces the old folder's Dice exactly on all 4536 rows, and
>   adds HD95). The script's `--verify` re-checks this for any future drop.
> - `synthseg_noEM` in the 2026-08-05 import was **truncated** (8-81 subjects per
>   test group instead of 27-219); now complete. Its combined Dice 50.4 -> 47.5.
> The superseded metrics are archived at `benchmark/03_archive/healthy-spine-tum_import_20260805/`.
> Still no SRCSM arm.


> ⚠️ **This dataset currently exists as `8_results_healthy-spine-tum/` ONLY.** No raw data,
> BIDS tree, nnUNet conversion, checkpoints, or split files live in this repo for this dataset —
> the metrics were computed on an external pipeline (not `run_job`/`00_commun_scripts`) and
> handed over as `tsv.zip`. `benchmark/validate_standard_dataset_structure.py` will correctly flag
> this dataset as missing slots 0–7 and 9 until/unless that data is onboarded here too.

## What this is

A 6-method comparison, trained on two contrasts (**CT** and **Dixon in-phase MRI**), evaluated
cross-contrast on two external spine test sets — **spider** (T1w, T2w) and **spinegan** (ct,
dixon-fat, dixon-inphase, dixon-water) — 3 folds each, mirroring this project's usual
train-one-modality / evaluate-cross-contrast methodology (see root `CLAUDE.md`).

## Method-name mapping (source dir → this project's roster)

| Source folder | This project's method | category | notes |
|---|---|---|---|
| `Base` | `baseline` | nnUNet | |
| `Paper` | `auglab_default` | auglab | |
| `Synthseg10NoEM+GE` | `synthseg_noEM` | auglab | |
| `Synthseg10+GE` | `synthseg_EM` | auglab | |
| `PALETTE-05+Paper` | `auglabAug_v26_6_2` (**OURS**) | auglab | user-confirmed 2026-08-05 |
| `PALETTE-05+GE` | `palette05_ge` | auglab | extra arm, **not** part of the fixed 6 — kept under its own name rather than force-mapped |

`palette05_ge`'s `eval_all.csv` files exist on disk under `8_results_healthy-spine-tum/02_metrics/`
but are intentionally excluded from every `06_evaluate/configs/*.yaml` — it never appears in the
per-modality, combined, or significance tables. This is a deliberate scope decision (not part of
the fixed 6/7-method suite), not an oversight.

No SRCSM arm is present in this drop — the 6-method table for this dataset is currently missing
that row (unlike the 4 core training datasets, which have all 6).

## Train-contrast mapping

`SG_CT` → `ct`, `SG_in-phase` → `inphase`.

## Metric conversion (source TSV → our `eval_all.csv`)

Source TSVs are panoptic-quality-style per-structure metrics (`vertebra`/`ivd`/`spinal_canal`),
not plain Dice/HD95. Per user decision (2026-08-05):
- `dice` = `<label>-global_bin_dsc` (whole-structure binary Dice — the closest analog to how
  "dice" is used elsewhere in this project; NOT the same as `sq_dsc`/`pq_dsc`, which are
  instance-matching-sensitive).
- `hd95` = `<label>-global_bin_hd95` since the 2026-10-02 drop; a non-finite/blank value is
  written as `nan` (same as `evaluate.py` for an empty mask).
- `group` in `eval_all.csv` = `<test_set>_<test_contrast>`, e.g. `spider_T1w`, `spinegan_dixon_fat`.

Conversion script (one-off, not part of the repo's script layers — see root CLAUDE.md's "put new
code in the right one" rule; kept here for provenance only, not intended to be re-run as-is):
`_import_tum_tsv_20260805.py` (in this dataset's root — deliberately outside the numbered
`5_scripts_*` convention since that skeleton doesn't exist for this dataset yet).
If more TUM spine batches arrive in this same TSV shape, rewrite this as a proper
`5_scripts_healthy-spine-tum/06_evaluate/06_00_import_tum_tsv.py` once the rest of the dataset
skeleton exists.

## Layout produced

```
8_results_healthy-spine-tum/
  01_predictions/   (empty — no prediction volumes in this drop, metrics only)
  02_metrics/healthy_spine_tum_model/{ct,inphase}/<category>_healthy-spine-tum_<contrast>_<method>_20260805_130846/fold{0,1,2}/eval_all.csv
```

Timestamp `20260805_130846` = import time (this repo never trained/predicted this data — there is
no real training/prediction timestamp to use).
