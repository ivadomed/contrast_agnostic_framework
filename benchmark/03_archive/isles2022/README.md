# isles2022 — ARCHIVED 2026-10-06 (kept as the scaffolding reference)

ISLES'22 acute/subacute ischemic stroke (Zenodo 10.5281/zenodo.7153326, 250 cases, usable 246), trained on DWI and FLAIR with the full 6-method suite + ladder rungs 2-5, evaluated on both.
Archived because it is **not a useful domain-generalization example** (DWI and FLAIR show the lesion with the same polarity, so the plain baseline transfers; the ADC positive control scored too low to show anything).
Full result, diagnostics and the lesson: the project notes, section "Stroke task: ISLES 2022 — ARCHIVED".

- **Do not run or "fix" the scripts in place.** They keep the `02_tasks/<task>/<dataset>` directory-depth hop counts ON PURPOSE: `benchmark/create_pipeline_scripts.py` clones them (default `--reference` = this dir) into
  `benchmark/02_tasks/<task>/<dataset>/`, where the hop counts are right. From `03_archive/` depth they would resolve wrongly.
- **License:** the zip's LICENSE forbids redistribution without the ISLES'22 team's written agreement: no git-annex, no public figures of these images.
- Contents: BIDS (all 3 contrasts, ADC under `derivatives/adc`), nnU-Net raw/preprocessed (Dataset140 DWI, Dataset141 FLAIR), splits (47 test / 199 pool, 3 folds), `5_scripts_isles2022/`, results (`8_results_isles2022/`: metrics, pins, predictions, best+final checkpoints of all 60 folds, `diag_case_level.md`).
- The ADC positive-control outputs live only on TamIA scratch (`$SCRATCH/isles2022/_diag_adc`); numbers are in the project notes.
