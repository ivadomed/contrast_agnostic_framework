# MS3SEG -- EVALUATION-ONLY dataset (MS lesion, tri-mask, Toshiba scanner)

> Evaluation only. No model is ever trained on it.

MS3SEG (Bawil et al., figshare 10.6084/m9.figshare.30393475.v6, CC-BY-4.0): 100
patients, Toshiba scanner (cross-vendor vs open-ms/mslesseg), FLAIR/T1WI/T2WI,
with tri-mask annotations distinguishing ventricles, normal white-matter
hyperintensities (nWMH), and MS lesions. Here it tests `open-ms` (both its
FLAIR- and T1w-trained families) on an independent cohort AND scanner vendor.

## Gotchas found before wiring anything up (verified empirically)

- **Broken NIfTI headers**: every source file has `scl_slope=NaN`, which makes
  nibabel's default scaled read return garbage (e.g. mask values
  `{0, 16448, 49087, 65535}` instead of the true raw uint8 `{0, 64, 191, 255}`).
  `00_utils/00_00_ingest_and_bidsify.py` reads via `dataobj.get_unscaled()` and
  rewrites a clean header.
- **Label identity is NOT small indices**: raw mask values are
  `0=background, 64=ventricles, 191=normal-WMH, 255=MS-lesion`. Confirmed by
  reorienting the combined 4-label mask and the archive's separate per-class
  binary masks (`abWMH`/`nWMH`/`Vent`) to a common orientation and checking
  spatial overlap -- NOT assumed from the folder/file names alone.
- **Image/mask orientation mismatch within the same archive**: images are LAS,
  the combined mask is RAS. Both get reoriented to the project's LPS by
  `03_preprocess/03_00_reorient_to_lps.py`, which also resolves the mismatch.
- **Thick-slice clinical acquisition**: 256x256x20 grid, 0.898x0.898x6.8mm --
  much more anisotropic than open-ms/mslesseg's near-isotropic research grids.
- **Cross-label-space scoring**: open-ms's model only predicts a binary
  `lesion` (id=1). Evaluation uses `--label_map '{"lesion": [1, 255]}'` to
  score MS-lesion only; ventricles/normal-WMH are excluded from scoring
  entirely (not counted as false positives against classes the model was
  never trained to predict).

## Pipeline

```
00_utils/00_00_ingest_and_bidsify.sh     # unscaled read + clean header, BIDSify
03_preprocess/03_00_reorient_to_lps.sh   # LAS/RAS -> LPS
05_predict/05_00_build_test_inputs.sh    # BIDS -> nnUNet imagesTs_{flair,t1w,t2w}/
05_predict/05_08_predict_openms_all.sh      # FLAIR-trained: 6 methods x 3 folds x 3 contrasts
05_predict/05_15_predict_openms_t1w_all.sh  # T1w-trained:   6 methods x 3 folds x 3 contrasts
06_evaluate/06_02_evaluate_all.sh        # evaluate FLAIR-trained
06_evaluate/06_03_evaluate_t1w_all.sh    # evaluate T1w-trained
06_evaluate/06_10_aggregate_from_config.sh configs/ms3seg_01_results.yaml
06_evaluate/06_10_aggregate_from_config.sh configs/ms3seg_t1w_01_results.yaml
```

Same section-14 cross-dataset shape as mslesseg (also -> open-ms); see
`datasets/STANDARDIZATION_CHECKLIST.md`.
