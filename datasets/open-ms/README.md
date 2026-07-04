# open-ms — brain MS lesion segmentation (texture-advantage experiment)

**Purpose.** A texture/spatial-structure-dominated task to test the hypothesis that the
image-driven **v26_6_2 + AugLab** method is *largely* (not incrementally) superior to the
label-driven **SynthSeg-EM** contender. MS lesions are small, focal, and texture/gradient-
defined — exactly where SynthSeg's per-region `mean + white-noise` generation (texture-blind)
should fail and v26's affine remap (texture-preserving) should win. See
`datasets/00_commun_scripts/00_04_analysis/README.md` for the mechanism + the existing-model
evidence that motivated this experiment.

## Data
`open_ms_data` (muschellij2/open_ms_data, cross-sectional; Lesjak et al. 2018, CC-BY).
30 patients, co-registered T1W/T2W/FLAIR (1mm iso, LPS, N4) + binary consensus lesion mask.
Pristine copy in `0_raw_open-ms/`.

## Design (mirrors CHAOS / BraTS: train one contrast, generalise cross-contrast)
- **Train on FLAIR** (single channel) with augmentation; **test cross-contrast** on
  FLAIR (in-domain) / T2W / T1W (lesions progressively more iso-intense → harder, more
  texture-dependent).
- **Split (patient-level, no leakage):** 22 train-pool (4-fold CV) + **8 held-out test**
  chosen stratified by lesion burden (`01_create_splits/01_01_create_splits.py`). The
  held-out 8 (lesion vox 330→52,706) are never in any fold; the trainer's `do_split`
  guard enforces this.
- nnUNet `Dataset070_OpenMS_FLAIR`. Methods: `auglabAug_v26_6_2_train025_val100` (OURS),
  `synthseg_EM_train100_val000` (contender); `baseline` + `auglab_default` scripts also
  provided. **2000 epochs, 4 folds.**

## Pipeline (run in order)
```bash
S=datasets/open-ms/5_scripts_open-ms
.venv/bin/python $S/00_utils/00_01_bidsify.py                 # 0_raw → 1_BIDS (hard-link + sidecars)
.venv/bin/python $S/01_create_splits/01_01_create_splits.py   # splits + held-out test (reads 0_raw masks)
.venv/bin/python $S/02_nnunet/02_00_convert.py                # 1_BIDS → Dataset070 (FLAIR tr + T2/T1 ts)
bash $S/03_preprocess/03_00_preprocess.sh                     # nnUNet plan&preprocess (CPU job) + install splits
bash $S/04_train/04_03_train_auglabAug_v26_6_2_train025_val100.sh   # OURS  (4 folds)
bash $S/04_train/04_02_train_synthseg_EM.sh                        # contender (4 folds)
```

## After training finishes — predict + evaluate
Current run IDs (all **6 methods** launched 2026-06-30, 2000 epochs × 4 folds, 60h walltime):
- **OURS (auglabAug_v26)**: `open-ms_flair_auglabAug_v26_6_2_train025_val100_20260630_064702` (auglab) — predict 05_03, eval CATEGORY=auglab
- **OURS (v26 alone)**:     `open-ms_flair_v26_6_2_train050_val100_20260630_102558` (nnUNet) — predict 05_07, eval CATEGORY=nnUNet
- synthseg_EM:    `open-ms_flair_synthseg_EM_train100_val000_20260630_064759` (auglab) — predict 05_02, eval CATEGORY=auglab
- synthseg_noEM:  `open-ms_flair_synthseg_noEM_train100_val000_20260630_102513` (auglab) — predict 05_06, eval CATEGORY=auglab
- auglab_default: `open-ms_flair_auglab_default_20260630_072742` (auglab) — predict 05_05, eval CATEGORY=auglab
- baseline:       `open-ms_flair_baseline_20260630_072657` (nnUNet) — predict 05_04, eval CATEGORY=nnUNet
```bash
S=datasets/open-ms/5_scripts_open-ms
bash $S/05_predict/05_03_predict_auglabAug_v26_6_2.sh  open-ms_flair_auglabAug_v26_6_2_train025_val100_20260630_064702 all
bash $S/05_predict/05_02_predict_synthseg_EM.sh        open-ms_flair_synthseg_EM_train100_val000_20260630_064759 all
bash $S/06_evaluate/06_01_evaluate_run.sh  open-ms_flair_auglabAug_v26_6_2_train025_val100_20260630_064702 auglab all
bash $S/06_evaluate/06_01_evaluate_run.sh  open-ms_flair_synthseg_EM_train100_val000_20260630_064759 auglab all
# the key texture-sensitive metric (lesion-wise detection, size-stratified, per contrast):
.venv/bin/python $S/06_evaluate/06_02_lesionwise_analysis.py \
    open-ms_flair_auglabAug_v26_6_2_train025_val100_20260630_064702 \
    open-ms_flair_synthseg_EM_train100_val000_20260630_064759 \
    --out datasets/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair/exp_texture_advantage/lesionwise.csv
```

## Expected result (hypothesis)
v26+auglab ≈ synthseg on in-domain FLAIR bulk Dice, but **increasingly superior on T2W→T1W
and on small-lesion detection / fewer catastrophic misses** — the texture-advantage
signature. (If results are *too* good or too bad, be suspicious and inspect predictions.)
