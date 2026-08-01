# KIDNEY-T2W — EVALUATION-ONLY dataset (T2w MRI kidney segmentation)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

KIDNEY-T2W ("T2-weighted Kidney MRI Segmentation", Zenodo record
[5153568](https://zenodo.org/records/5153568), CC-BY-4.0, open access) is a
**T2-weighted MRI, kidney-only** dataset. Here it serves a single purpose: **test
models trained on `chaos` (MR T1-DUAL in-phase / T2-SPIR) on an independent MRI
cohort/scanner**, measuring MR→MR domain-randomization generalization on the
**kidney** label. This is the second independent MRI test set beyond `chaos`
itself (after `cirrmri-liver`, which covers liver) and the first for the kidney
organ.

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_kidney-t2w/` subdirs are intentionally empty.
- The pipeline only **consumes** chaos-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the chaos model folders, and `06_evaluate` scores the
  results against this dataset's kidney ground truth.

## Data

- **100 T2w MRI volumes**: 50 Healthy Control (`HC`) + 50 Chronic Kidney Disease
  (`CKD`) subjects. 10 subjects (5 HC, 5 CKD) were scanned repeatedly (up to 5
  scans each, for precision assessment) — each repeat is a separate case here.
  BIDS subject labels: `sub-HC{NN}`/`sub-CKD{NN}`, with an `r{K}` suffix for
  repeat scans (e.g. `sub-HC26r3`).
- Binary kidney masks (both kidneys merged into a single label), already
  reasonably tight around the kidney region.
- Raw NIfTI orientation is **RSP** (a coronal-acquisition convention — the thin
  axis is A-P, not S-I), reoriented to the project's canonical **LPS** via the
  shared `orient.py` (lossless axis permute/flip, no resampling).

## Evaluation specifics

- chaos models emit 4 labels (1=liver, 2=R-kidney, 3=L-kidney, 4=spleen). This
  dataset's GT is kidney-only, stored as a single **GT label 1** (both kidneys
  merged). Evaluation uses a **custom evaluator shim**
  (`06_evaluate/06_00_evaluate_kidney_t2w.py`, mirrors the `trusted` dataset's
  pattern exactly) that calls `eval_metrics.run_evaluation` directly with
  `pred_id=[2, 3]` (union of chaos's two kidney ids) vs `gt_id=1` — this needs the
  list-valued `pred_id` merge, which the generic `evaluate.py --label_map` CLI
  (single-int only) cannot express.
- CHAOS has a restricted axial FOV (see `datasets/chaos/README` / the shared
  `00_00_utils/fov.py`); this dataset's full-abdomen T2w volumes are FOV-restricted
  to the CHAOS-equivalent slab, anchored on the **kidney** (GT id 1) — reusing the
  existing `kidney` anchor in `chaos_fov_margins.json` (no new anchor needed; the
  `trusted` dataset already established it).
- Single input channel (T2w) matches the chaos T2spir model most closely, but both
  chaos T1in- and T2spir-trained models are evaluated (cross-modality both ways).
- **Methods evaluated: 7 runs** (not the full 8-run roster used by other cross-eval
  sets) — `baseline`, `auglab_default`, `synthseg_noEM`, `synthseg_EM`, `srcsm`, and
  OURS `auglabAug_v26_6_2_train050_val000`/`val100`. The standalone `v26_6_2` (alone,
  pre-AugLab-wrapper) arm is **deliberately excluded** here per an explicit scoping
  decision for this dataset — see `06_evaluate/06_02_evaluate_all_t1in.sh` /
  `06_03_evaluate_all_t2spir.sh`.
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).

## License note

Released under **CC-BY-4.0** (attribution only, no share-alike restriction),
Zenodo record 5153568. Cite the associated dataset record.
