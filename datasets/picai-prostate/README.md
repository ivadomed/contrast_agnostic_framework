# picai-prostate — clinically significant prostate cancer (csPCa) lesion segmentation

**Role: TRAINING dataset** (6-method suite × 2 training modalities × 3 folds), like
`open-ms`, `chaos`, `brats2024-glioma` and `on-harmony` — not an eval-only cohort.

## Why this dataset

We needed a second **intra-tissue, texture-defined** segmentation target: one where the
structure to segment has **no visible interface** with what surrounds it. A csPCa lesion is
a subtle intensity/texture change *inside* otherwise normal-looking prostate parenchyma —
no capsule, no edge, no anatomical boundary to latch onto. Radiologists find it by T2
hypointensity plus diffusion restriction, not by tracing a contour.

That places it on the **texture-defined** side of the causal-ablation ladder, alongside
open-ms (MS lesions in white matter) and brats2024-glioma (tumour sub-regions inside
brain), and opposite chaos / msd-spleen / sliver07 (boundary-defined organs). It is the
first **non-brain** texture-defined task in the suite, so it tests whether the
texture-preservation result generalises beyond neuro.

The three contrasts are also genuinely different regimes — anatomical T2W, a *quantitative
diffusion parameter map* (ADC), and a heavily diffusion-weighted image (HBV) — which
stretches the cross-contrast generalisation axis further than the within-family contrasts
of the brain datasets.

## Source

| | |
|---|---|
| Images | PI-CAI Challenge, Public Training and Development Dataset — [Zenodo 6624726](https://doi.org/10.5281/zenodo.6624726), CC-BY-NC-4.0. 1500 bpMRI studies, 1476 patients, 3 Dutch centres (Radboudumc, UMCG, ZGT), 2012–2021. |
| Labels | [`DIAGNijmegen/picai_labels`](https://github.com/DIAGNijmegen/picai_labels) — **human-expert** csPCa lesion delineations (`csPCa_lesion_delineations/human_expert/resampled/`). AI-derived delineations (Bosma22a) are deliberately **not** used. |
| Prostate masks | `anatomical_delineations/whole_gland/AI/Bosma22b/` — used *only* to centre the resampling grid, never as a label. |
| Reference | Saha et al., PI-CAI Challenge (2022). |

Download: `bash 5_scripts_picai-prostate/00_utils/00_00_download.sh` (~27 GB; login-node
only — tamia's compute nodes have no outbound network).

## Case selection — 219 studies

| | count |
|---|---|
| csPCa-positive **with** human-expert delineation → **kept** | **219** |
| csPCa-negative (all-zero mask) → dropped | 1075 |
| positive but AI-annotated only (no human-expert file) → dropped | 205 |
| lesion outside the prostate-centred slab → dropped | 1 |
| total | 1500 |

Two deliberate exclusions:

- **Negatives are dropped.** Keeping them would leave ~70 % of the test set with an empty
  ground truth, where per-case Dice is undefined and would dominate the headline number.
  So the task here is *"segment the lesion in a known-positive exam"* — stated up front
  rather than buried in a denominator.
- **AI-annotated positives are dropped.** Mixing expert and model-generated ground truth
  into one benchmark makes a segmentation score hard to interpret. 219 human-expert cases
  is in the same range as open-ms (30), chaos (~40) and on-harmony (~30).

Split: **55 test patients / 164 train-pool patients**, patient-level (a patient's studies
never straddle a boundary), test set stratified by lesion burden so it spans small→large
lesions. 4-fold CV file, folds **0/1/2** trained (CLAUDE.md fold policy).

## Common grid — the one non-obvious preprocessing step

PI-CAI ships each study as three series on three different grids (T2W ~0.3×0.3×3.0 mm,
ADC/HBV ~2×2×3.0 mm). The project's cross-contrast protocol needs **one mask valid for
every test contrast**, so `00_01_bidsify.py` resamples all three series *and* the label
onto one reference grid per study:

    spacing (0.5, 0.5, 3.0) mm · size (256, 256, 24) · T2W direction cosines
    centred on the prostate (AI whole-gland centroid; image centre as fallback)

This is deliberately the geometry PI-CAI's own baseline uses (`picai_prep`
`matrix_size=[20,256,256]`, `spacing=[3.0,0.5,0.5]`), widened 20→24 slices. The series
share a scanner frame of reference, so resampling by physical coordinates co-registers
them — no image-based registration is performed or needed. Verified: all three contrasts
and the mask come out with byte-identical affines.

Resampling T2W 0.3→0.5 mm is also what makes nnU-Net's 3d_fullres patches affordable; at
native resolution the in-plane matrix is ~2× larger per axis for no extra information at
lesion scale.

## Layout

```
0_raw_picai-prostate/     images/<pid>/<pid>_<sid>_{t2w,adc,hbv}.mha  +  picai_labels/
1_BIDS_picai-prostate/picai-prostate-bpmri/
    sub-<pid>/ses-<sid>/anat/sub-<pid>_ses-<sid>_{T2w,ADC,HBV}.nii.gz
    derivatives/manual_masks/sub-<pid>/ses-<sid>/anat/..._dseg.nii.gz
    cases.json                      case id -> {patient, study, n_lesion_voxels}
2_nnUNet_picai-prostate/raw/
    Dataset080_PICAI_T2W/           training modality 1  (imagesTr + imagesTs_{t2w,adc,hbv})
    Dataset081_PICAI_ADC/           training modality 2  (same three test-contrast dirs)
4_splits_picai-prostate/            partition.json, splits_final.json, test_cases.json
```

HBV never gets its own `Dataset0xx` — it is test-only, so it appears solely as
`imagesTs_hbv/labelsTs_hbv` inside both training datasets. Case id is
`<patient_id>_<study_id>`.

**On tamia all bulk paths move to `$SCRATCH`** via `scripts/cluster/tamia_env_picai.sh`
(its `/project` is at its file-count quota). Only the repo, the splits and the venv live in
`/project`. Scratch is purgeable — `00_00_download.sh` + the 00→03 pipeline reproduces
everything.

## Pipeline

```bash
bash 5_scripts_picai-prostate/00_utils/00_00_download.sh            # login node, ~27 GB
bash 5_scripts_picai-prostate/00_utils/00_01_bidsify.sh            # -> common grid
bash 5_scripts_picai-prostate/01_create_splits/01_01_create_splits.sh
bash 5_scripts_picai-prostate/02_nnunet/02_00_convert.sh           # Dataset080 (T2W)
bash 5_scripts_picai-prostate/02_nnunet/02_01_convert_adc.sh       # Dataset081 (ADC)
bash 5_scripts_picai-prostate/03_preprocess/03_00_preprocess.sh
bash 5_scripts_picai-prostate/03_preprocess/03_01_preprocess_adc.sh
# training — tamia (whole-node H100 packs, all 36 fold-trainings):
bash 5_scripts_picai-prostate/04_train/04_21_tamia_pack_launch_all.sh
# training — one-sbatch-per-fold clusters (vulcan/killarney/romane):
bash 5_scripts_picai-prostate/04_train/04_07_run_all_t2w.sh
bash 5_scripts_picai-prostate/04_train/04_17_run_all_adc.sh
```

## The method suite — 6 runs, 7 arms

6 method wrappers per modality × 3 folds × 2 modalities = **36 fold-trainings**, producing
the **7** headline arms. The sixth method uses the **DualVal** trainer: one training run
emits both `auglabAug_v26_6_2_train050_val000` (clean validation) and `..._val100`
(synth-only validation), hard-linked as two ordinary predict-ready RUN_IDs. That is the
project standard since 2026-07-25 and is selection-equivalent to training the two
separately — **do not add a seventh training run.**

Predict wrappers for *both* `_val000` and `_val100` must set
`TRAINER=nnUNetTrainerPICAIProstateAugLabDualVal`; both mirrors live under that trainer's
directory name.

Epochs: **2000** (matching on-harmony / open-ms).
