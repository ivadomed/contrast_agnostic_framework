# TRUSTED — EVALUATION-ONLY dataset (kidney CT + 3D ultrasound)

> ⚠️ **Evaluation only. No model is ever trained on it.**
> ⚠️ **Redistribution-restricted** — IRCAD France TRUSTED Data Use Agreement
> (`0_raw_trusted/data_use_agreement.txt`). Do not re-upload or share the data.

TRUSTED (IRCAD France) is a paired **kidney CT ↔ 3D ultrasound** registration
dataset. Here it serves one purpose: **test models trained on `chaos` (MR T1-DUAL
in-phase / T2-SPIR) on out-of-modality CT *and* US kidney data**, measuring
MR→{CT,US} domain-randomization generalization. It is the first **ultrasound**
dataset in the codebase and the first eval dataset with **two test modalities**.

## Why no training here

- There is **no `01_create_splits` or `04_train`** stage — those `5_scripts_trusted/`
  subdirs are intentionally empty.
- The pipeline only **consumes** chaos checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the chaos model folders; `06_evaluate` scores the
  results against the TRUSTED kidney ground truth.

## Data (what we ingest)

From the archive (`TRUSTED_dataset_for_nsd.zip`, figshare file 51079133) we ingest
images + the **estimated-GT** masks; meshes, landmarks and registration transforms
are skipped (irrelevant to segmentation eval).

- **CT**: 48 contrast-enhanced abdominal volumes, **both kidneys** per volume.
- **US**: 59 3D ultrasound volumes, **one kidney each** (left/right encoded in the
  filename/`acq-` entity; 48 patients, US is a paired subset).
- Each TRUSTED mask is a **single binary kidney label**. GT masks ship as `float32`
  with floating-point junk values; ingest **thresholds (>0.5) and casts to uint8**.
- Orientation is already `('L','P','S')` (the chaos/sliver07 convention) — no
  reorientation needed; `03_preprocess/03_00_check_orientation.py` verifies this
  idempotently.

## Evaluation specifics

- chaos models emit 4 labels (1=liver, 2=R-kidney, 3=L-kidney, 4=spleen). TRUSTED GT
  is a single binary kidney. The evaluator (`06_00_evaluate_trusted.py`) **merges
  chaos {2,3} → "kidney"** and scores that vs GT(1). This does not penalize L/R
  confusion — in a cropped US field-of-view side is ambiguous; the question is
  whether the kidney is segmented at all under domain shift.
- Single input channel (CT or US) matches the chaos single-channel MR model. nnUNet
  applies chaos's MR normalization to the CT/US input — intended; this is exactly
  what the synthesis / domain-randomization models are built to absorb.
- Both contrast branches are evaluated: **t1in** (`env.sh` default) and **t2spir**
  (pre-export `CHAOS_TRAINING_CONTRAST=t2spir`, or source `00_utils/env_t2spir.sh`).

## Pipeline

```
00_utils/00_00_ingest_and_bidsify.sh    # extract CT+US, binarize masks → BIDS
03_preprocess/03_00_check_orientation.sh # idempotent LPS check (no-op here)
05_predict/05_00_build_test_inputs.py    # BIDS → nnUNet imagesTs_{ct,us}/ (+labels)
05_predict/05_07_predict_chaos_all.sh    # t1in: all methods × 4 folds × {ct,us}
05_predict/05_15_predict_chaos_t2spir_all.sh   # t2spir variant
06_evaluate/06_02_evaluate_all_chaos.sh  # evaluate t1in + t2spir (kidney)
06_evaluate/06_03_aggregate_results.sh   # comparison tables + heatmaps
```

This is the same §14 cross-dataset shape as `amos` and `sliver07`; see
`datasets/STANDARDIZATION_CHECKLIST.md`.
