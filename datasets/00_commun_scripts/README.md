# `datasets/00_commun_scripts/` — shared pipeline drivers

Every dataset's pipeline (train → predict → evaluate, plus splits/convert) runs on the
**same shared drivers** that live here. A dataset does not copy logic — it writes a few
tiny **shims** that set dataset-specific values and then hand off to the shared driver.

> **Why this exists:** before this, each of `04_00_common.sh`, the predict-common script,
> and the evaluators were copy-pasted per dataset and drifted apart. Now there is one
> implementation; a dataset is just *configuration*. Adding a new dataset is filling in
> blanks, not rewriting a pipeline — and `validate_standard_dataset_structure.py` +
> `STANDARDIZATION_CHECKLIST.md` catch you if you stray.

This is **not** a dataset — it is skipped by the validator (see `IGNORED_ENTRIES`).

```
00_commun_scripts/
  00_00_utils/
    common_env.sh          # env loader: derives all standard paths from $DATASET_NAME
    splits_lib.py          # kfold_splits(), write_splits_final()
    nnunet_convert_lib.py  # gzip_copy(), run_threaded_conversion(), write_dataset_json()
    eval_metrics.py        # dice_score(), hd95(), run_evaluation()  (label-triple based)
    eval_aggregate.py      # load_run(), cross_fold_stats(), report + heatmap builders
  00_01_train/
    train_common.sh        # per-fold nnU-Net training driver (resume, GPU pinning, run_job)
  00_02_predict/
    predict_common.sh      # nnU-Net prediction driver (own-model AND cross-dataset modes)
  00_03_evaluate/
    evaluate.py            # method-agnostic Dice/HD95 evaluator (CLI)
    summarize_fold.py      # merge per-group CSVs → eval_all.csv + eval_summary.md
    aggregate_results.py   # cross-fold/-experiment report + heatmaps
```

> Directory names have leading digits, so they are **not** importable as Python packages.
> Shared modules are imported by inserting the dir on `sys.path`; module **file** names
> therefore avoid leading digits (`eval_metrics.py`, not `00_eval.py`).

---

## How a dataset plugs in (the shim model)

```
5_scripts_<name>/04_train/04_07_train_baseline.sh   ← method wrapper (sets METHOD/TRAINER…)
        └─ sources 04_train/04_00_common.sh          ← SHIM (sets DATASET_ID_DEFAULT…)
                └─ sources 00_commun_scripts/00_01_train/train_common.sh   ← SHARED DRIVER
```

The method wrappers and the shim keep their normal names and CLIs. Only the *body* of the
shim changed: it sets a handful of variables and `source`s the shared driver. The same
shape repeats for `00_utils/env.sh`, `05_predict/*common.sh`, and the evaluators.

---

## Adding a new dataset — step by step

Use an existing dataset as the template:

| You're building… | Copy the shims from | Notes |
|---|---|---|
| a dataset you train on AND predict with its own models | **chaos** | the canonical full pipeline |
| a multi-contrast training dataset | **brats2024-glioma** | several `TRAINING_CONTRAST`s, `env_<contrast>.sh` overrides |
| a **test-only** dataset (run *another* dataset's models on it) | **sliver07** (single modality) or **amos** (multi-modality, label remap) | uses predict `cross` mode + the `CHAOS_*` env block |

### 0. Scaffold + validate
```bash
cd datasets && .venv/bin/python create_dataset_structure.py   # prompts for <name>; makes the 9 empty subdirs
.venv/bin/python datasets/validate_standard_dataset_structure.py   # run after every step
```
The scaffolder only creates the directory tree (incl. `8_results/{01_predictions,02_metrics}`)
— you fill in the `5_scripts_<name>/` shims below. Walk `STANDARDIZATION_CHECKLIST.md` top to
bottom; it maps 1:1 to the steps here.

### 1. `00_utils/env.sh` — the dataset "config"
Set the dataset-specific values, then source `common_env.sh`. It derives `nnUNet_raw`,
`PREDICTIONS_ROOT`, `METRICS_ROOT`, `WANDB_PROJECT`, `PYTHONPATH`, etc. from `$DATASET_NAME`.
```bash
#!/usr/bin/env bash
DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="<name>"
export NNUNET_DATASET_ID="Dataset0NN_<Name>"
export MODEL_TYPE="<name>_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1n}"   # conditional → env_<contrast>.sh can override

BIDS_SUBDIR="<name>-<region>"          # → BIDS_ROOT (omit if 1_BIDS has no leaf dir)
CE_SUBDIRS="raw preprocessed splits"   # which optional standard dirs this dataset has
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# Anything common_env doesn't own (conditional so contrast overrides survive a re-source):
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_<name>/01_predictions/<name>_model/t1n/nnUNet}"
```
**Contract** (`common_env.sh`): you MUST have set `DATASET_ROOT` and `DATASET_NAME` first.
Optional control vars (plain, not exported): `DATASET_ROLE` (default `training`),
`BIDS_SUBDIR`, `CE_SUBDIRS` ⊆ `{raw preprocessed splits}`, `CE_EXTRA_PYTHONPATH`.

### 2. `01_create_splits/01_01_create_splits.py` — use `splits_lib`
Do your dataset-specific discovery + hold-out, then call the shared k-fold + writer:
```python
import sys; from pathlib import Path
DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from splits_lib import kfold_splits, write_splits_final

pool = discover_and_holdout(...)                 # YOUR logic (seed-shuffle, sealed test set)
splits = kfold_splits(pool, n_folds=4)           # shared: subject-level CV chunking
write_splits_final(splits, SPLITS_DIR, NNUNET_PRE, NNUNET_DATASET)  # writes + copies to preprocessed
```

### 3. `02_nnunet/02_00_convert.py` — use `nnunet_convert_lib`
Provide your per-case `convert_fn(case) -> case_id` and your dataset.json fields:
```python
from nnunet_convert_lib import gzip_copy, run_threaded_conversion, write_dataset_json
case_ids = run_threaded_conversion(cases, lambda c: convert_case(c, images_tr, labels_tr), jobs=16)
write_dataset_json(out_dir, channel_names={"0": "..."}, labels={...},
                   num_training=len(case_ids), name="...", description="...", reference="...")
```
Also add `02_nnunet/<Name>Trainers.py` — a registration shim that imports your trainer
classes (copy chaos's `CHAOSTrainers.py`; the `*Trainers.py` name is **required** by
nnU-Net's class discovery and is copied into the venv — see CLAUDE.md).

### 4. `04_train/04_00_common.sh` — shim over `train_common.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
DATASET_ID_DEFAULT="0NN"
NNUNET_NUM_EPOCHS_DEFAULT="500"
# NNUNET_ITERS_PER_EPOCH_DEFAULT="150"   # optional — omit to keep nnU-Net's built-in default
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_01_train/train_common.sh" "$@"
```
Method wrappers (`04_0X_train_<method>.sh`) set `METHOD`, `TRAINER`, `DA_WORKERS`, `LOG_DIR`
(and optionally `NNUNET_NUM_EPOCHS`, `nnUNet_compile`, `GPUS_PER_FOLD`, `FOLD_SLOT_GPU`,
`SINGLE_FOLD`), then `source ../04_train/04_00_common.sh "$@"`. Launch:
```bash
bash 04_07_train_baseline.sh                 # fresh run (auto RUN_ID)
bash 04_07_train_baseline.sh <RUN_ID>        # resume / extend (raise NNUNET_NUM_EPOCHS to extend)
```

### 5. `05_predict/05_01_predict_common.sh` — shim over `predict_common.sh`
**Own-model** dataset:
```bash
PREDICT_MODE="own"
PREDICT_JOB_PREFIX="<name>_predict";  PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="t1n t1c t2w t2f"        # contrasts/modalities to predict
PREDICT_FOLD_DEFAULT="all";  PREDICT_DATASET_ID_DEFAULT="0NN"
PREDICT_TIME="3:00:00";  PREDICT_EXTRA_FLAGS=""
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
```
**Cross-dataset** (test-only, runs e.g. chaos models — see sliver07/amos): set
`PREDICT_MODE="cross"`, `PREDICT_EXTRA_FLAGS="-npp 12 -nps 6"`, and the `CHAOS_*` block in
`env.sh`. The driver then reads the model from `CHAOS_PREDICTIONS_ROOT`, inputs from this
dataset's flat `imagesTs_<item>/`, and writes outputs under your `PREDICTIONS_ROOT`.
```bash
bash 05_03_predict_<method>.sh <RUN_ID>            # all folds (default), all default items
bash 05_03_predict_<method>.sh <RUN_ID> 2 t1n      # fold 2, t1n only
```

### 6. `06_evaluate/` — `06_01_evaluate_run.sh` + the shared evaluator
`06_01_evaluate_run.sh` (per dataset) calls, per fold per group:
```bash
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/evaluate.py" \
    --pred_dir D --gt_dir D --out_csv C --name <group> \
    --dataset_json J [--labels liver]          # same-label-space; OR:
    --label_map map.json                         # cross-space {name:[pred_id,gt_id]} (e.g. amos)
```
then merges the per-group CSVs + writes the summary with the shared summarizer:
```bash
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
    "$EVAL_DIR" "$RUN_ID" "$F" --group-col modality --groups-word Modalities --groups "${groups[@]}"
```
and aggregates across folds/experiments:
```bash
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_results.py" \
    --metrics_dir <METRICS_ROOT/...> --title "<NAME>" [--run_keys ...] [--fold0-heatmap]
```

**Standard `06_01` CLI (all datasets):** `bash 06_01_evaluate_run.sh <RUN_ID> [FOLD] [items…]`.
`CATEGORY` (`nnUNet`|`auglab`) is an **env override** (`CATEGORY=auglab bash 06_01 …`) or is
**auto-detected** from `RUN_ID` — never a positional argument.

---

## Driver contracts (quick reference)

**`00_01_train/train_common.sh`** — shim sets `DATASET_ID_DEFAULT`,
`NNUNET_NUM_EPOCHS_DEFAULT`, opt. `NNUNET_ITERS_PER_EPOCH_DEFAULT`; wrapper sets `METHOD`,
`TRAINER`, `DA_WORKERS`, `LOG_DIR`; opt. `NNUNET_RESULTS_BASE`, `GPUS_PER_FOLD` (1 or 2/DDP),
`NNUNET_NUM_EPOCHS`, `nnUNet_compile`, `SINGLE_FOLD`/`SINGLE_SLOT`/`SINGLE_GPU`,
`FOLD_SLOT_GPU`, `RESUME_WANDB_IDS`, `LAUNCH_WAIT`. `$1` = `RUN_ID` (resume).

**`00_02_predict/predict_common.sh`** — shim sets `PREDICT_MODE` (`own`|`cross`),
`PREDICT_JOB_PREFIX`, `PREDICT_LOG_PREFIX`, `PREDICT_ITEMS_DEFAULT`, `PREDICT_FOLD_DEFAULT`,
`PREDICT_DATASET_ID_DEFAULT` (own only; cross uses `CHAOS_DATASET_ID`), `PREDICT_TIME`
(`""` → omit `--time`), `PREDICT_EXTRA_FLAGS`. Wrapper sets `METHOD`, `TRAINER`, `CATEGORY`.

**`00_03_evaluate/evaluate.py`** — `--pred_dir --gt_dir --out_csv [--name] [--workers]`
plus labels via `--dataset_json [--labels …]` **or** `--label_map <file|json>`. Model- and
dataset-agnostic: it scores `(label_name, pred_id, gt_id)` triples (cross-label-space eval
is just `pred_id != gt_id`).
**`summarize_fold.py`** — `<eval_dir> <run_id> <fold> --groups … [--group-col --groups-word
--label-word --title-suffix --note]`. Writes the canonical `eval_all.csv` (consumed by the
aggregators) + a uniform `eval_summary.md`.
**`aggregate_results.py`** — `--metrics_dir --title [--run_keys …] [--fold0-heatmap]`.

**`00_00_utils/splits_lib.py`** — `kfold_splits(pool, n_folds)`,
`write_splits_final(splits, splits_dir, nnunet_pre, dataset_name)`.
**`nnunet_convert_lib.py`** — `gzip_copy(src, dst)`,
`run_threaded_conversion(items, convert_fn, jobs, progress_every=0)`,
`write_dataset_json(out_dir, channel_names, labels, num_training, **extra)`.

---

## Conventions the validator enforces (so a new dataset can't drift)

- **9-subdir layout** `0_raw`…`9_tests`; scripts under `5_scripts_<name>/` are numbered
  dirs `NN_name/` containing `NN_NN_name.ext` files (`00_utils/` and `*Trainers.py` exempt).
- **Results layout:** `8_results_<name>/01_predictions/` and `02_metrics/` are **required**;
  `03_aggregated_results/` optional. `01_predictions/` has one subdir per model family
  (`nnUNet/`, `auglab/`, …) — never run IDs at the top level.
  *(on-harmony is the lone exception — still on the legacy `01_results/`+`02_nnUNet_results/`
  layout, migration pending.)*
- **`env.sh`** sets `DATASET_NAME` = dir name, exports `WANDB_PROJECT=mri_synthesis_seg_${DATASET_NAME}`
  (via common_env — no bare `mri_synthesis_seg`), `PREDICTIONS_ROOT`/`METRICS_ROOT` under `8_results`.
- **Run-naming:** `{dataset}_{contrast}_{method}_{YYYYMMDD_HHMMSS}` (auto-generated by the
  train driver).

Run `validate_standard_dataset_structure.py` — it should print `✓` for your dataset.

---

## Stays dataset-specific / not consolidated (by design)

- **Per-case convert logic, split discovery/hold-out, dataset.json fields** — inherently
  dataset-specific; the shared libs cover only the mechanical parts.
- **Cross-dataset knowledge** — AMOS `ORGAN_MAP`, CHAOS per-modality `--labels`, SLIVER07
  liver-only — lives in each dataset; the evaluator mechanism is shared.
- **`*Trainers.py`** registration shims and **download/bidsify** scripts (heterogeneous I/O:
  Kaggle/Zenodo, DICOM/.mhd/.nii) — left per-dataset.
- The amos/sliver07 `06_03` aggregators keep a distinct organ-/liver-centric `00_comparison.md`
  layout but import the shared `load_run`/`cross_fold_stats` primitives.
