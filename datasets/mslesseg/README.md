# MSLesSeg -- EVALUATION-ONLY dataset (brain MS lesion, independent cohort)

> Evaluation only. No model is ever trained on it.

MSLesSeg (Guarnera, Rondinella et al., University of Catania; figshare
10.6084/m9.figshare.27919209.v1, CC-BY-4.0) is a brain MRI Multiple Sclerosis
lesion segmentation dataset: 115 scans from 75 patients, FLAIR/T1/T2, with
expert-validated binary lesion masks. Here it serves one purpose: **test models
trained on `open-ms`** (brain MS lesion segmentation, FLAIR-trained) on an
**independent cohort / scanner set**, measuring MS-lesion domain-randomization
generalization beyond the dataset the model was trained on.

## Why no split

- There is **no `01_create_splits` or `04_train`** stage -- those
  `5_scripts_mslesseg/` subdirs are intentionally empty.
- The archive itself ships a train/test split (53 patients x up to 3
  timepoints = 93 scans in `train/`, 22 single-timepoint scans in `test/`), but
  since we never train here, that split is meaningless for us. Per project
  decision, **all 115 scans are collapsed into ONE flat evaluation pool** (see
  `00_utils/00_00_ingest_and_bidsify.py`) -- the archive's `source_split`
  provenance is kept only as a `participants.tsv` column, not used to
  partition anything.
- The pipeline only **consumes** open-ms checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the open-ms model folders; `06_evaluate` scores
  the results against MSLesSeg's own lesion ground truth.

## Data (what we ingest)

From `MSLesSeg Dataset.zip` (figshare file 52771814) we take the preprocessed
scans + lesion masks. We do NOT use `MSLesSeg_RAW.zip` (raw DICOM-derived
NIfTI, pre-registration) or `MSLesSeg-2024-main.zip` (the authors'
preprocessing code) -- not needed for a pure eval-only consumer of already
co-registered data.

- Each scan ships as `{FLAIR,T1,T2,MASK}.nii.gz`, already co-registered to
  FLAIR, skull-stripped, and resampled to 1mm-iso MNI152 space
  (182,218,182) -- no further geometric preprocessing needed.
- MASK is already binary `uint8`-valued `{0,1}` (single lesion label) -- no
  float32-junk gotcha like TRUSTED's kidney masks.
- **Orientation is LAS as shipped** (verified via `nib.aff2axcodes`), NOT the
  project's LPS convention (the chaos/sliver07/open-ms family) -- unlike
  TRUSTED (already LPS), this dataset needs a REAL reorientation, same
  situation as AMOS. `03_preprocess/03_00_reorient_to_lps.py` performs the
  lossless axis permute/flip.

## Evaluation specifics

- open-ms's ground truth label space (`background: 0, lesion: 1`) matches
  MSLesSeg's directly -- **no cross-label-space merge needed** (unlike
  TRUSTED's chaos-kidney {2,3}->1 union). `06_00_evaluate_mslesseg.py` is a
  thin shim over the shared `evaluate.py`, same as open-ms's own evaluator.
- open-ms trains a **single** model (FLAIR only, unlike chaos's two training
  contrasts t1in/t2spir) and tests cross-contrast on FLAIR/T2W/T1W. MSLesSeg
  mirrors that: all three of its contrasts (flair, t1w, t2w) are fed to the
  same open-ms FLAIR-trained checkpoints as cross-contrast generalization
  probes.
- No FOV restriction (brain, not abdomen -- unlike TRUSTED's CHAOS-FOV CT
  restriction).

## Pipeline

```
00_utils/00_00_ingest_and_bidsify.sh     # extract archive, BIDSify, hard-link (train+test -> 1 pool)
03_preprocess/03_00_reorient_to_lps.sh   # LAS -> LPS (real reorient, lossless)
05_predict/05_00_build_test_inputs.sh    # BIDS -> nnUNet imagesTs_{flair,t1w,t2w}/ (+labels)
05_predict/05_08_predict_openms_all.sh   # 6 methods x 3 folds x {flair,t1w,t2w}
06_evaluate/06_02_evaluate_all.sh        # evaluate all 6 methods
06_evaluate/06_10_aggregate_from_config.sh configs/mslesseg_01_results.yaml
```

This is the same section-14 cross-dataset shape as `amos`/`sliver07`/`trusted`;
see `datasets/STANDARDIZATION_CHECKLIST.md`. The one structural generalization
this dataset required: `00_commun_scripts/00_02_predict/predict_common.sh`'s
`cross` mode used to hardcode `CHAOS_*` env var names -- it now reads a
`SOURCE_PREFIX` var (defaults to `CHAOS` for existing datasets), and
mslesseg's `05_01_predict_common.sh` sets `SOURCE_PREFIX=OPENMS` to point it at
`OPENMS_*` vars instead, without duplicating the driver.
