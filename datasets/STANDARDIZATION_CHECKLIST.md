# Dataset Pipeline Standardization Checklist

Use this checklist when onboarding a new dataset or auditing an existing one.
Each section maps to a concrete part of the pipeline. Check every item against
the actual files — do not rely on memory or the fact that a similar dataset
already passed.

---

## 1. Directory structure

- [ ] Exactly nine numbered subdirectories exist:
  `0_raw_<dataset>`, `1_BIDS_<dataset>`, `2_nnUNet_<dataset>`,
  `3_conf_<dataset>`, `4_splits_<dataset>`, `5_scripts_<dataset>`,
  `6_checkpoints_<dataset>`, `7_analysis_<dataset>`, `8_results_<dataset>`,
  `9_tests_<dataset>`
- [ ] `datasets/validate_standard_dataset_structure.py` passes cleanly for
  this dataset
- [ ] `8_results_<dataset>/` has the three sub-result dirs:
  `01_predictions/`, `02_metrics/`, `03_aggregated_results/`
- [ ] `01_predictions/` has one subdir per model family (`nnUNet/`, `auglab/`,
  …) — never run IDs directly at the top level

---

## 2. `00_utils/env.sh`

- [ ] `DATASET_NAME` is set and matches the directory name exactly
  (e.g. `export DATASET_NAME="chaos"`)
- [ ] `DATASET_ROLE` is set to one of: `training | test-only | both`
  and a comment explains what each value means
- [ ] `WANDB_PROJECT` is exported as `mri_synthesis_seg_${DATASET_NAME}` — no
  hardcoded string, and no stale `mri_synthesis_seg` without a suffix
- [ ] `PREDICTIONS_ROOT` points to `8_results_<dataset>/01_predictions`
- [ ] `METRICS_ROOT` points to `8_results_<dataset>/02_metrics`
- [ ] `nnUNet_results` (for training datasets) points to
  `8_results_<dataset>/01_predictions/nnUNet`
- [ ] `PYTHONPATH` exports the `5_scripts_<dataset>` dir so the dataset's
  Python package is importable
- [ ] For **test-only / both** datasets: cross-dataset source vars are present
  (`<SOURCE>_DATASET_ROOT`, `<SOURCE>_PREDICTIONS_ROOT`, `<SOURCE>_NNUNET_RAW`,
  `<SOURCE>_DATASET_ID`, `<SOURCE>_DS_NAME`, `<SOURCE>_DATASET_JSON`)

---

## 3. Run naming convention

Every run ID must follow exactly:
```
{dataset_name}_{synth_approach}_train{XXX}_val{YYY}_{YYYYMMDD_HHMMSS}
```

- [ ] `04_00_common.sh` auto-generates via
  `RUN_ID="${1:-${DATASET_NAME}_${METHOD}_$(date +%Y%m%d_%H%M%S)}"` — the
  `${DATASET_NAME}_` prefix is present
- [ ] Each `04_0X_train_*.sh` sets `METHOD` with synth-prob suffixes:
  - `_train090_val100` (90% train, 100% val synthesis)
  - `_train100_val000` (synthseg / auglab: always 100% train, 0% val)
  - `_train050_val100`, `_train025_val000`, … as appropriate
  - Methods without synthesis (e.g. `auglab_default`, baselines) need no
    probability suffix
- [ ] No train script passes a bare, prefix-less RUN_ID explicitly (e.g.
  `"v26_6_2_${TS}"`) — if passing explicitly, the full convention name is used
  (e.g. `"chaos_v26_6_2_train090_val100_${TS}"`)
- [ ] All batch-launch scripts (`04_1X_launch_*.sh`) that pass existing RUN_IDs
  as defaults use the current canonical names

---

## 4. Existing run directories

### `01_predictions/`
- [ ] Every dir under `nnUNet/` and `auglab/` starts with `{dataset_name}_`
- [ ] Every dir encodes synth probs where applicable (`_train{XXX}_val{YYY}`)
- [ ] No stale dirs with old short names remain (e.g. `v26_6_2_20260607_001859`,
  `synthseg_EM`, `auglab_default`)

### `02_metrics/`
- [ ] Every dir follows `{category}_{RUN_ID}` where `RUN_ID` includes the
  dataset prefix (e.g. `nnUNet_chaos_v26_6_2_train090_val000_20260614_205937`)
- [ ] Each metrics dir has a matching predictions dir (no orphaned metrics)

---

## 5. Trainer Python package

- [ ] `5_scripts_<dataset>/` is a valid Python package (importable via
  `PYTHONPATH`)
- [ ] Package root `__init__.py` exists with the standard boilerplate comment
  (not empty — has `__all__: list[str] = []` and a comment pointing to the
  nnUNet registration shim)
- [ ] `trainers/` subdir exists (may be a symlink to `02_nnunet/trainers/`)
- [ ] `trainers/__init__.py` exists with the standard boilerplate comment
- [ ] The nnUNet registration shim
  (`.venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/<Dataset>Trainers.py`)
  exists and imports every concrete trainer class used by this dataset
- [ ] After adding a new trainer, the shim import is added and importable:
  `.venv/bin/python -c "from <pkg>.trainers.<module> import <Class>"`

---

## 6. Trainers

- [ ] Every trainer inherits from the correct MRO:
  `Dataset binding → Method base (V26_6_x / SynthSeg / AugLab) → nnUNetTrainerFast → nnUNetTrainer`
- [ ] `train_synth_prob` and `val_synth_prob` are set where synthesis is
  probabilistic; SynthSeg trainers have neither (deterministic, always 100%)
- [ ] `nnUNetTrainerFast._save_run_config()` is called in `initialize()` (base
  class handles this — verify the base class still calls it after any refactor)
- [ ] `_run_config_extras()` is overridden in subclasses that have extra
  config fields (auglab JSON paths, etc.)
- [ ] WandB image logging uses a correct mid-slice convention for the dataset:
  - BraTS: `[:, :, mid]` (axial, last dim)
  - CHAOS: verify slice axis matches anatomy

---

## 7. `04_train/04_00_common.sh`

- [ ] `RUN_ID` auto-generation includes `${DATASET_NAME}_` prefix
- [ ] `nnUNet_wandb_project` uses `${WANDB_PROJECT:-mri_synthesis_seg_${DATASET_NAME}}`
  — not a hardcoded string
- [ ] `AUGLAB_VAL_PARAMS_GPU_JSON` is exported (even if empty) so the env var
  is always defined inside the systemd slice
- [ ] `DATASET_ID` default matches the actual nnUNet dataset number for this
  dataset
- [ ] `PYTHONPATH` inside the `set_slot` subprocess includes the scripts dir
  so trainers are importable without installing the package

---

## 8. Training scripts (`04_train/04_0X_*.sh`)

- [ ] Every training script has a clear `# Usage:` comment with:
  - How to start a new run (no explicit RUN_ID → auto-generates)
  - How to resume (pass the full canonical RUN_ID)
  - No stale "pass RUN_ID explicitly so wandb name is exactly X" language
    from the old convention
- [ ] `LOG_DIR` is under `/tmp/nnunet_<dataset>_<method>/` (not clashing
  across methods)
- [ ] `NNUNET_RESULTS_BASE` is set for auglab runs (pointing to
  `01_predictions/auglab`) — nnUNet runs do not override it (use env default)
- [ ] `FOLD_SLOT_GPU` or `SINGLE_FOLD` are set correctly where applicable

---

## 9. Predict scripts (`05_predict/05_0X_*.sh`)

- [ ] Every predict script has a correct `# Example:` comment that uses a
  current canonical RUN_ID (with dataset prefix and prob encoding)
- [ ] Default `RUN_ID` values (e.g. `${1:-chaos_v26_6_2_...}`) point to dirs
  that actually exist in `01_predictions/`
- [ ] `CATEGORY` (`nnUNet` vs `auglab`) is set correctly and matches where the
  run dir actually lives
- [ ] `TRAINER` class name matches the class registered in the shim
- [ ] For **test-only / both** datasets: predict scripts that run foreign models
  set `nnUNet_results` to the source dataset's predictions dir, not this
  dataset's own dir

---

## 10. Evaluate scripts (`06_evaluate/`)

Script numbering convention (match CHAOS as the reference):
```
06_00_evaluate.py          — core evaluate function (Python)
06_01_evaluate_run.sh      — evaluate a single run (all folds or one fold)
06_02_aggregate_results.py — cross-run aggregation + heatmaps
06_02_aggregate_results.sh — shell wrapper that calls the Python script
06_03_predict_eval_all_exps.sh   — batch: predict + evaluate all experiments
06_04_predict_eval_batch2.sh     — batch: second experiment batch
06_05_predict_eval_batch3.sh     — batch: third experiment batch (folds 2/3)
```

- [ ] Script names follow the numbered convention above
- [ ] `06_01_evaluate_run.sh` usage header says `06_01_evaluate_run.sh`, not
  an old renamed version
- [ ] Example RUN_IDs in `06_01_evaluate_run.sh` use current canonical names
- [ ] `06_02_aggregate_results.py` docstring example uses `--run_keys` (not
  `--run_ids`) and shows a current canonical `{category}_{RUN_ID}` key
- [ ] All batch scripts (`06_03`, `06_04`, `06_05`) with hardcoded or default
  RUN_IDs use canonical names with dataset prefix and prob encoding
- [ ] For **test-only / both** datasets: evaluate scripts for foreign models
  read from `01_predictions/{source_dataset}_models/` or equivalent namespacing
  so foreign and own predictions never collide

---

## 11. Aggregate results script (`06_02_aggregate_results.py`)

- [ ] Dice values displayed as percentage without "%" sign (multiply by 100,
  1 decimal place): `0.8786 → 87.9`
- [ ] HD95 values shown with 1 decimal place: `12.345 → 12.3`
- [ ] Heatmap y-axis (run names) has reduced font size (`labelsize=7`) to
  accommodate long canonical run names
- [ ] `fmt_cell()` uses the percentage convention for Dice
- [ ] `build_modality_summary()` summary table uses the same convention
- [ ] `_save_heatmaps()` cell text uses the same convention
- [ ] `build_fold0_heatmap()` (if present) uses the same convention

---

## 12. Resume scripts

- [ ] Any explicit-resume script (e.g. `resume_*_to_N.sh`) references the
  correct `01_predictions/nnUNet/<canonical_RUN_ID>` path — not `02_nnUNet_results/`
  or any other stale path prefix
- [ ] WandB project in the resume script uses the per-dataset project name
  (`mri_synthesis_seg_<dataset>`)
- [ ] WandB run name in the resume script includes the dataset prefix

---

## 13. WandB

- [ ] All training runs in WandB are under project `mri_synthesis_seg_<dataset>`
  (check the WandB UI — old runs in the shared project cannot be moved, but note
  the discrepancy)
- [ ] WandB run names follow `{RUN_ID}_fold{k}` so they sort alongside the run
  directory name

---

## 14. Cross-dataset evaluation (for `DATASET_ROLE=both`)

When a dataset evaluates models trained on another dataset:

- [ ] Predictions from foreign models live under
  `01_predictions/{source_dataset}_models/{category}/{RUN_ID}/` — not mixed
  with own predictions at the top level
- [ ] Metrics from foreign models live under
  `02_metrics/{source_dataset}_models_{category}_{RUN_ID}/` — clearly namespaced
- [ ] Dedicated predict scripts exist for each foreign model family
  (e.g. `05_0X_predict_chaos_<method>.sh`)
- [ ] Dedicated evaluate scripts exist for each foreign model family
  (e.g. `06_0X_evaluate_chaos_models.sh`)
- [ ] Aggregate script handles own vs. foreign runs separately or labels them
  clearly in the report

---

## 15. `run_config.json` (auto-saved by `nnUNetTrainerFast`)

After a training run starts, each fold directory should contain:

- [ ] `run_config.json` with at minimum:
  - `trainer_class`, `fold`, `num_epochs`, `run_id`, `dataset_name`
  - `train_synth_prob`, `val_synth_prob` (if applicable)
  - `auglab_params_json`, `auglab_val_params_json` (if applicable)
- [ ] The `run_id` field in the JSON matches the directory name

---

## 16. Quick commands to verify a dataset

```bash
# 1. Structure
.venv/bin/python datasets/validate_standard_dataset_structure.py <dataset>

# 2. Trainer importable
.venv/bin/python -c "import <pkg>.trainers"

# 3. Prediction dirs follow convention
ls datasets/<dataset>/8_results_<dataset>/01_predictions/nnUNet/
ls datasets/<dataset>/8_results_<dataset>/01_predictions/auglab/

# 4. Metrics dirs follow convention and are 1-to-1 with predictions
ls datasets/<dataset>/8_results_<dataset>/02_metrics/

# 5. No stale short names (should print nothing)
ls datasets/<dataset>/8_results_<dataset>/01_predictions/nnUNet/ \
   datasets/<dataset>/8_results_<dataset>/01_predictions/auglab/ \
   | grep -v "^${DATASET_NAME}_"

# 6. env.sh sanity
(source datasets/<dataset>/5_scripts_<dataset>/00_utils/env.sh && \
 echo "DATASET_NAME=$DATASET_NAME" && \
 echo "DATASET_ROLE=$DATASET_ROLE" && \
 echo "WANDB_PROJECT=$WANDB_PROJECT")
```
