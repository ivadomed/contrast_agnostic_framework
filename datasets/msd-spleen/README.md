# MSD-SPLEEN — EVALUATION-ONLY dataset (CT spleen segmentation)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

MSD-SPLEEN (Medical Segmentation Decathlon, Task09_Spleen) is a **CT, spleen-only**
dataset. Here it serves a single purpose: **test models trained on `chaos` (MR
T1-DUAL in-phase / T2-SPIR) on out-of-modality CT data**, measuring MR→CT
domain-randomization generalization on the **spleen** label — the one CHAOS organ
(liver/kidney/spleen) with no dedicated cross-eval set before this (`sliver07`
covers liver, `trusted` covers kidney).

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_msd-spleen/` subdirs are intentionally empty.
- The pipeline only **consumes** chaos-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the chaos model folders, and `06_evaluate` scores the
  results against the MSD-SPLEEN spleen ground truth.

## Data

- **41 labeled CT volumes** (the Decathlon's "training" split) — used as **our test
  set**. Source NIfTI (`imagesTr`/`labelsTr`) from the msd-for-monai S3 mirror
  (`https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar`, confirmed
  public/unauthenticated, ~1.5 GB, CC-BY-SA 4.0). No WAF/DUA gate, unlike TRUSTED's
  figshare download.
- The Decathlon's **20 unlabeled "test" volumes are NOT used** (no public ground
  truth) — not extracted, mirroring the chaos/sliver07 sealed-test policy.
- Case numbering is non-contiguous (MSD reserves some ids for the unlabeled split);
  BIDS subject labels are `sub-SP{NN}` using the original MSD case number.

## Evaluation specifics

- chaos models emit 4 labels (1=liver, 2=R-kidney, 3=L-kidney, 4=spleen). MSD-SPLEEN
  GT is spleen-only, stored as **GT label 1** (see `00_00_download_and_bidsify.py`).
  Because chaos's spleen id (4) and this dataset's GT id (1) don't match numerically
  (unlike sliver07's liver, which happened to be id 1 on both sides), evaluation uses
  the shared evaluator's **`--label_map '{"spleen": [4, 1]}'`** cross-label-space
  mode — the same mechanism `trusted` uses to merge chaos's two kidney ids into its
  single kidney GT.
- CHAOS has a restricted axial FOV (see `datasets/chaos/README` / the shared
  `00_00_utils/fov.py`); MSD-SPLEEN full-torso CT is FOV-restricted to the
  CHAOS-equivalent slab, anchored on the **spleen** (this dataset's only organ).
  `datasets/chaos/5_scripts_chaos/06_evaluate/06_30_measure_chaos_fov.py` was
  extended with a `spleen` anchor (chaos GT id 4) to measure this margin.
- Single input channel (CT) matches the chaos in-phase model. nnUNet applies chaos's
  MR normalization to the CT input — intended; this is exactly what the synthesis /
  domain-randomization models are built to absorb.
- Methods evaluated: the full current 8-run roster (`baseline`, `v26_6_2` alone,
  `synthseg_EM`, `synthseg_noEM`, `auglab_default`, `srcsm`, and OURS
  `auglabAug_v26_6_2_train050_val000`/`val100`, per contrast) — see
  `06_evaluate/06_02_evaluate_all_t1in.sh` / `06_03_evaluate_all_t2spir.sh`.
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).

## License note

Medical Segmentation Decathlon data is released under **CC-BY-SA 4.0** (share-alike,
attribution). Cite Antonelli et al., *Nature Communications* 13, 4128 (2022).
