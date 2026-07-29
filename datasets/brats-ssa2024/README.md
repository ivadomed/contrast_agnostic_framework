# BraTS-SSA 2024 -- EVALUATION-ONLY dataset (glioma, Sub-Saharan Africa cohort)

> Evaluation only. No model is ever trained on it.

BraTS-SSA (Sub-Saharan Africa) is the pre-treatment adult glioma segmentation task
from the 2024 Brain Tumor Segmentation (BraTS) challenge family -- the largest public
collection of annotated pretreatment glioma scans from African patients. Official
distribution is via Synapse (synapse.org/brats) and requires registration + a Data Use
Agreement; there is no open direct-download link. Here it serves one purpose: **test
models trained on `brats2024-glioma`** (both its T1n- and T2w-trained families) on an
**independent cohort / scanner / population**, measuring glioma-segmentation
domain-randomization generalization beyond the adult-glioma cohort the models were
trained on.

## Data source (why kagglehub, not Synapse)

Ingested from the kagglehub public re-upload `kaalmurlidhar/brats2024-africa`
(anonymous download, no credentials needed) rather than Synapse directly, per explicit
user instruction. That archive bundles TWO cohorts:
- `95_Glioma/` (95 cases) -- **used here**.
- `51_OtherNeoplasms/` (51 cases, non-glioma tumor types) -- **NOT used**. brats2024-glioma
  is a glioma-specific segmenter; including other neoplasm types would be out-of-domain
  and would confound the comparison. (Spot-checked: one OtherNeoplasms case's GT was
  missing label 2/SNFH entirely, consistent with a different pathology.)

## Label compatibility (verified empirically before onboarding)

- `brats2024-glioma` (Dataset051/052) uses brats2024-glioma's post-treatment 5-class
  scheme: `background=0, NCR=1, SNFH=2, ET=3, RC=4` (RC = resection cavity, post-op).
- BraTS-SSA is **pre-treatment** (pre-operative scans) -- its GT has only 3 foreground
  labels: `NCR=1, SNFH=2, ET=3`. **RC=4 never appears** (verified: `np.unique` on the raw
  seg.nii files for multiple glioma cases showed only `{0,1,2,3}`).
- The label IDs that DO overlap match exactly (1/2/3 mean the same thing on both
  sides) -- no cross-label-space merge/remap needed, just a **subset restriction**
  (`--labels NCR SNFH ET`, excluding RC from scoring entirely rather than counting it
  as false-positive against a class that structurally cannot exist here). Simpler than
  TRUSTED's kidney merge, more careful than mslesseg/open-ms's exact 1:1 match.

## Data details

- Each case ships as `BraTS-SSA-<id>-000/BraTS-SSA-<id>-000-{t1c,t1n,t2f,t2w,seg}.nii`
  (uncompressed `.nii`), standard BraTS grid (240x240x155, 1mm iso).
- **Orientation is RAS as shipped** (verified via `nib.aff2axcodes`), NOT the project's
  LPS convention -- same situation as AMOS/MSLesSeg. `03_preprocess/03_00_reorient_to_lps.py`
  performs the lossless axis permute/flip.

## Evaluation specifics

- brats2024-glioma trains **two separate model families** (T1n-trained, Dataset051;
  T2w-trained, Dataset052) -- like chaos's t1in/t2spir split. BraTS-SSA is cross-evaluated
  against **both**, each with its own 6-method suite and aggregate config
  (`configs/brats-ssa_t1n_01_results.yaml` / `_t2w_01_results.yaml`).
  `00_utils/env_t2w.sh` switches the T2w branch (mirrors trusted's env_t2spir.sh).
- The T2w `auglabAug_v26_6_2` method uses the **dual-val trainer**
  (`nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal`) for BOTH its train050_val000 and
  train050_val100 variants -- both mirrors live under the same trainer/RUN_ID timestamp
  (verified on disk, not assumed) -- see [[project_dualval_trainer]]. T1n's two variants
  are regular separate single-checkpoint runs (different TRAINER, different timestamps).
- All four BraTS-SSA contrasts (t1n/t1c/t2w/t2f) are fed to both trained-model families
  as cross-contrast generalization probes, mirroring brats2024-glioma's own held-out test.

## Pipeline

```
00_utils/00_00_ingest_and_bidsify.sh     # gzip + BIDSify (95_Glioma cohort only)
03_preprocess/03_00_reorient_to_lps.sh   # RAS -> LPS (real reorient, lossless)
05_predict/05_00_build_test_inputs.sh    # BIDS -> nnUNet imagesTs_{t1n,t1c,t2w,t2f}/ (+labels)
05_predict/05_14_predict_brats_t1n_all.sh   # T1n-trained: 6 methods x 3 folds x 4 contrasts
05_predict/05_15_predict_brats_t2w_all.sh   # T2w-trained: 6 methods x 3 folds x 4 contrasts
06_evaluate/06_02_evaluate_t1n_all.sh    # evaluate all 6 T1n-trained methods
06_evaluate/06_03_evaluate_t2w_all.sh    # evaluate all 6 T2w-trained methods
06_evaluate/06_10_aggregate_from_config.sh configs/brats-ssa_t1n_01_results.yaml
06_evaluate/06_10_aggregate_from_config.sh configs/brats-ssa_t2w_01_results.yaml
```

Same section-14 cross-dataset shape as amos/sliver07/trusted/mslesseg; see
`datasets/STANDARDIZATION_CHECKLIST.md`. Uses `predict_common.sh`'s `SOURCE_PREFIX`
indirection (`SOURCE_PREFIX=BRATS`) -- no driver changes needed, that generalization
was already made for mslesseg.
