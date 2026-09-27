# SLIVER07 — EVALUATION-ONLY dataset

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

SLIVER07 (MICCAI 2007 liver segmentation challenge) is a **CT, liver-only** dataset.
Here it serves a single purpose: **test models trained on `chaos` (MR T1-DUAL in-phase)
on out-of-modality CT data**, measuring MR→CT domain-randomization generalization.

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_sliver07/` subdirs are intentionally empty.
- The pipeline only **consumes** chaos-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the chaos model folders, and `06_evaluate` scores the
  results against the SLIVER07 liver ground truth.

## Data

- **20 labeled CT volumes** (the challenge's "training" set) — used as **our test set**.
  Source `.mhd/.raw` MetaImage in `0_raw_sliver07/{scan,label}/`.
- The challenge's **10 unlabeled "test-scans" are NOT used** (no public ground truth) —
  not downloaded, mirroring the chaos sealed-test policy.

## Evaluation specifics

- chaos models emit 4 labels (1=liver, 2=R-kidney, 3=L-kidney, 4=spleen).
  SLIVER07 GT is liver only, so **evaluation binarizes predictions to label 1 (liver)**.
- Single input channel (CT) matches the chaos in-phase model. nnUNet applies chaos's
  MR normalization to the CT input — intended; this is exactly what the synthesis /
  domain-randomization models are built to absorb.
- Methods evaluated: `baseline` (MR-only control, expected to fail on CT), `v26_6_2`,
  `synthseg_EM`, `synthseg_noEM`, `auglab_default`.

## License note

SLIVER07 challenge rules forbid reporting liver-segmentation results on training data
alone for leaderboard submission. Our use is an **internal cross-modality robustness
test** (no training, no leaderboard claim). Cite Heimann et al., IEEE TMI 28(8), 2009.
