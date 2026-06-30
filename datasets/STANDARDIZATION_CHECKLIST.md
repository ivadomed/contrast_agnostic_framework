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
- [ ] `8_results_<dataset>/` has the two REQUIRED sub-result dirs
  `01_predictions/` and `02_metrics/` (validator enforces these);
  `03_aggregated_results/` is optional. (All datasets are now on this layout —
  the legacy `01_results/` + `02_nnUNet_results/` trees, formerly used by
  on-harmony, were migrated/archived on 2026-06-29.)
- [ ] `01_predictions/` has one subdir per model family (`nnUNet/`, `auglab/`,
  …) — never run IDs directly at the top level. For a **cross-dataset eval
  dataset** there is an extra `<model_type>/<contrast>/` level above the family
  dirs — see §14.

---

## 1b. Image geometry / orientation  ⚠️ critical for cross-dataset eval

A model applied to a new dataset sees the **voxel array**, and nnU-Net/SimpleITK
honour the affine — but any step that compares in array space (and visual QA)
breaks if the new dataset's stored orientation differs from the datasets the
model was trained/validated on. AMOS shipped CT as `LAS` and MRI as `RAS` while
chaos/sliver07 are `LPS`, so predictions came out Y-flipped until reoriented.

- [ ] Every image's stored orientation matches the reference datasets'
  canonical convention. Check with
  `nib.aff2axcodes(nib.load(f).affine)` — for the chaos/sliver07 family this is
  `('L', 'P', 'S')`. Do this for **images and segmentation masks**, all modalities.
- [ ] If orientation differs, reorient **losslessly** (axis permute/flip only —
  `nibabel.as_reoriented` / `ornt_transform`, NOT resampling) so it is label-safe
  and image+mask stay aligned. Sanity-check anatomy after (e.g. spine posterior).
- [ ] `0_raw_<dataset>/` is left **pristine** — reorientation writes to BIDS /
  nnUNet trees only (temp-file + atomic replace if `1_BIDS` hard-links `0_raw`).
- [ ] Reorientation is a numbered, idempotent pipeline step (e.g.
  `03_preprocess/03_00_reorient_to_lps.py`), not a one-off — and BIDSify notes it.
- [ ] Voxel spacing / intensity ranges are sane vs the reference dataset
  (CT in HU, MRI arbitrary — matters for any intensity-based step).

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
  is always defined inside the job environment
- [ ] `DATASET_ID` default matches the actual nnUNet dataset number for this
  dataset
- [ ] `PYTHONPATH` passed into the job includes the scripts dir so trainers are
  importable without installing the package
- [ ] **Jobs are dispatched via `run_job`, never `set_slot`/raw `sbatch`.** On
  Vulcan `run_job` auto-routes to Slurm (`scripts/job_runner/`); the old
  `set_slot`/systemd-slice workstation path is dead. Pipeline scripts must stay
  backend-neutral (see project `CLAUDE.md`).

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

Script roles (match CHAOS as the reference — exact numbers vary by dataset):
```
06_00_evaluate*.py             — core evaluate function (Dice/HD95)
06_01_evaluate_run.sh          — evaluate a single run (all folds or one fold)
06_0X_evaluate_all_*.sh        — run 06_01 for every run (per contrast/source);
                                 reads run IDs from a config, runs them in PARALLEL
06_03_aggregate_results.{py,sh}  — single-dataset cross-run aggregation + heatmaps
06_10_aggregate_from_config.sh — config-driven aggregator (shared:
                                 scripts/evaluate/aggregate_from_config.py) — the
                                 mechanism behind cross-dataset roll-ups
configs/*.yaml                 — per-contrast / cross-dataset aggregation configs
```

- [ ] `06_01_evaluate_run.sh` usage header names itself correctly (not an old
  renamed version) and its example RUN_IDs are current canonical names
- [ ] Aggregation scripts `source env.sh` then `cd "${PROJECT_ROOT}"` — **no
  hardcoded absolute paths** (a `cd /home/<old-host>/...` line silently breaks
  the script on a different machine; this happened in AMOS `06_03`)
- [ ] `_all_` / batch scripts submit method×fold jobs **in parallel** (background
  each method + `wait`), not one method at a time — there is no reason to gate
  one independent job on another on Slurm
- [ ] For **test-only / cross-dataset** datasets: evaluate scripts for foreign
  models read from `01_predictions/<model_type>/<contrast>/...` (see §14) so
  foreign and own predictions never collide

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

## 14. Cross-dataset / test-only eval dataset (e.g. evaluating CHAOS models)

This is the shape AMOS and SLIVER07 use. Read this section start-to-finish when
adding a new dataset that exists to evaluate **foreign (chaos-trained) models**.

### Layout — note the `<model_type>/<contrast>/` levels
Predictions and metrics are namespaced by the **source-dataset model family** and
the **training contrast**. General form (chaos is the only current instance):
```
01_predictions/{source_dataset}_model/<contrast>/<category>/<RUN_ID>/fold{k}/<test_item>/
02_metrics/{source_dataset}_model/<contrast>/<category>_<RUN_ID>/fold{k}/eval_*.csv
```
where `{source_dataset}_model` is the value of `<SOURCE>_MODEL_TYPE` — e.g.
`chaos_model` (note: **singular** `_model`, not `_models`); `<contrast>` ∈
{`t1in`, `t2spir`, …}; `<category>` ∈ {`nnUNet`, `auglab`}.

- [ ] Predictions/metrics follow the general form above (NOT run IDs at the top
  level, and the `<contrast>` level is present — earlier drafts of this checklist
  omitted it). If `DATASET_ROLE=both`, the dataset's **own** models live under a
  different `<model_type>` (e.g. `<this_dataset>_model`) so own vs. foreign
  predictions never collide.

### env.sh cross-dataset source vars  (general `<SOURCE>_*`; chaos shown as the instance)
- [ ] `<SOURCE>_MODEL_TYPE` (e.g. `CHAOS_MODEL_TYPE=chaos_model`),
  `<SOURCE>_DATASET_ROOT`, `<SOURCE>_PREDICTIONS_ROOT`, `<SOURCE>_NNUNET_RAW`
  are set.
- [ ] **Per-contrast** vars are switchable: `<SOURCE>_TRAINING_CONTRAST`,
  `<SOURCE>_DATASET_ID`, `<SOURCE>_DS_NAME`, `<SOURCE>_DATASET_JSON` default to
  `t1in` and are overridden by a sibling `env_<contrast>.sh` (e.g.
  `env_t2spir.sh` pre-exports them, then re-sources `env.sh`). Mirror this for
  any additional contrast.
- [ ] `CE_EXTRA_PYTHONPATH` adds the source dataset's scripts dir so the foreign
  trainer classes resolve (no new trainers/shim needed for a pure eval dataset).

### Per-contrast predict + evaluate
- [ ] Dedicated predict scripts per foreign method, **per contrast**
  (`05_0X_predict_chaos_<method>.sh`, `05_1X_predict_chaos_t2spir_<method>.sh`),
  plus an `_all` wrapper per contrast that fans them out **in parallel**.
- [ ] Predict scripts set `nnUNet_results` to the **source** model dir (via the
  `CHAOS_*` vars), not this dataset's own.
- [ ] `_all` evaluate wrapper per contrast (e.g. `06_0X_evaluate_all_t2spir.sh`)
  pre-exports the contrast vars and runs `06_01` for each run in parallel.

### Cross-dataset roll-up (the combined heatmaps)
- [ ] For the new dataset to appear in the **combined** cross-dataset tables, add
  it as a `source` (with `column_prefix`/`column_rename`) **and** to
  `column_order` in BOTH
  `chaos/.../06_evaluate/configs/cross_dataset_t1in_01_results.yaml` and
  `cross_dataset_t2spir_01_results.yaml`. Forgetting this is silent — the
  aggregation just omits your columns.
- [ ] After eval, re-run `06_10_aggregate_from_config.sh <cross_dataset_*.yaml>`
  to refresh `chaos/.../02_metrics/chaos_model/<contrast>/04_01_cross_dataset_*`.

### Re-running after the source/data changes
- [ ] If the new dataset's **images change** (e.g. a reorientation fix, §1b),
  ALL contrast branches that consume them (t1in *and* t2spir) are stale and must
  be re-predicted + re-evaluated + re-aggregated — not just the one you touched.

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

# 7. Orientation matches the reference family (§1b) — should all print ('L','P','S')
.venv/bin/python - <<'PY'
import glob, nibabel as nib
for f in glob.glob("datasets/<dataset>/1_BIDS_<dataset>/**/*.nii.gz", recursive=True)[:20]:
    print(nib.aff2axcodes(nib.load(f).affine), f)
PY

# 8. Cross-dataset eval dataset is wired into the combined roll-ups (§14)
grep -l "<dataset>" \
  datasets/chaos/5_scripts_chaos/06_evaluate/configs/cross_dataset_*.yaml
```
