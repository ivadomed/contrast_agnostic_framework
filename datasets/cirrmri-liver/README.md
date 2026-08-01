# CIRRMRI-LIVER — EVALUATION-ONLY dataset (cirrhotic liver T1w+T2w MRI)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

> 🚫 **EXCLUDED from the cross-dataset roll-up tables as of 2026-08-01 —
> results here should not be trusted.** See
> `CIRRHOSIS_FRAGMENTATION_INVESTIGATION.md` for the full writeup. Summary:
> predictions on a subset of cases show liver/kidney/spleen labels fragmented
> into many small disconnected components with organs mislabeled/missing,
> across every method tested (baseline, srcsm, OURS) — not explained by any
> pipeline mechanism checked (FOV, orientation, intensity/normalization,
> trainer-class code, patch-tiling, config override, resampling/
> interpolation — all ruled out with direct empirical tests, not assumption).
> Leading unconfirmed explanation: this is a diagnosed cirrhosis cohort
> (Radiological Evaluation metadata: Mild/Moderate/Severe), and portal
> hypertension commonly causes splenomegaly (spleen approaching liver size) —
> an anatomical distortion no augmentation strategy here randomizes. Not
> independently confirmed (this dataset's own GT is liver-only, no spleen/
> kidney label to check against).

CIRRMRI-LIVER (CirrMRI600+, Jha et al., *Scientific Data* 12, 896 (2025)) is an
**MRI, liver-only** dataset. Here it serves a single purpose: **test models trained
on `chaos` (MR T1-DUAL in-phase / T2-SPIR) on out-of-distribution MRI data** — a
different scanner/protocol/patient-cohort (cirrhotic liver disease) than CHAOS's own
training data — measuring **MRI→MRI** domain-randomization generalization. It is the
first genuinely independent MRI cross-eval set added beyond AMOS's own MRI arm, and
completes the CT-side liver coverage (`sliver07`) with an MRI counterpart.

## Why this dataset (and not the Duke Liver/Spleen MRI sets)

Duke's Liver (DLDS) and Spleen (DSDS) MRI datasets were considered first but require
a **human-approval email request** to Duke (`DOCR.Help@dm.Duke.edu`, CC BY-NC-ND
license — no-derivatives) before any download is possible — not something that can
be completed autonomously. CirrMRI600+ is hosted on OSF with
`access_requests_enabled=false` (fully public, no gate) under **CC BY-NC 4.0**
(permits redistribution/adaptation, non-commercial only), confirmed via the OSF v2
API before committing to it.

## Why no training here

- There is **no `01_create_splits`, `04_train`** stage — those `5_scripts_cirrmri-liver/`
  subdirs are intentionally empty.
- The pipeline only **consumes** chaos-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the chaos model folders, and `06_evaluate` scores the
  results against the CirrMRI600+ liver ground truth.

## Data

- **337 unique patients** (310 with a T1w volume, 318 with a T2w volume, 291 with
  both) — the dataset's own train/valid/test split is **ignored and combined**
  (we don't train here, so that split has no meaning; every case is part of our
  single test set).
- Source: `https://osf.io/cuk24/` (public, no access request). Only the two 3D zips
  (`Cirrhosis_T1_3D.zip`, `Cirrhosis_T2_3D.zip`) are used — `Cirrhosis_T2_2D.zip`
  (2D slices, not usable by a 3d_fullres nnU-Net model) and `Healthy_subjects.zip`
  (no cirrhosis-specific value for a segmentation probe) are intentionally skipped.
- **Download quirk**: Vulcan's compute-node outbound proxy returns `403 Forbidden`
  for `osf.io` (unlike S3, which MSD-Spleen's download used successfully) — so
  `00_00_download.py` **must run directly on the login node** (a network fetch, not
  compute — negligible CPU/RAM, fits the login-node exception), not via `run_job`.
  `00_01_bidsify.py` (real computation) still runs via `run_job` as usual.
- Images/masks already stored in the project's canonical **LPS** orientation and
  binary `{0,1}` masks — no reorientation or float-junk cleanup needed (unlike
  AMOS/msd-spleen and TRUSTED respectively). `03_00_check_orientation.py` verifies
  this idempotently anyway, per the standardization checklist.
- Case ids are the original CirrMRI600+ numeric patient ids; BIDS subject labels
  `sub-CR{NNN}` zero-pad them. Two test items per subject: `t1` (T1w) and `t2`
  (T2w) — not every subject has both.

## Evaluation specifics

- chaos models emit 4 labels (1=liver, 2=R-kidney, 3=L-kidney, 4=spleen).
  CIRRMRI-LIVER GT is liver-only, stored as **GT label 1 — the same numbering
  chaos itself uses** (unlike msd-spleen, where chaos's spleen id (4) didn't match
  the dataset's own GT id (1) and needed a `--label_map` cross-space remap). This
  dataset uses the simpler `--dataset_json --labels liver` mode, like sliver07.
- CHAOS has a restricted axial FOV; CIRRMRI-LIVER's full-extent MRI volumes are
  FOV-restricted to the CHAOS-equivalent slab, anchored on the **liver** — reusing
  chaos's existing liver FOV margins (no new anchor needed, unlike msd-spleen which
  needed a new spleen anchor added to `chaos_fov_margins.json`).
- Methods evaluated: the full current 8-run roster (`baseline`, `v26_6_2` alone,
  `synthseg_EM`, `synthseg_noEM`, `auglab_default`, `srcsm`, and OURS
  `auglabAug_v26_6_2_train050_val000`/`val100`, per contrast) — see
  `06_evaluate/06_02_evaluate_all_t1in.sh` / `06_03_evaluate_all_t2spir.sh`.
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).
- Predict+evaluate ran on TamIA via a bounded 4-way worker-pool pack job (same
  pattern as msd-spleen's `05_20_tamia_pack_predict_evaluate.sh`, with every lesson
  from that onboarding — env-sourcing order, true bounded concurrency, exact
  row-count verification (not existence/exit-code) — applied from the start).

## License note

CirrMRI600+ is released under **CC BY-NC 4.0** (non-commercial). Cite: Jha, D.,
Susladkar, O. K., Gorade, V., et al. (2025). Large Scale MRI Collection and
Segmentation of Cirrhotic Liver. *Scientific Data*, 12(1), 896.
https://doi.org/10.1038/s41597-025-05201-7
