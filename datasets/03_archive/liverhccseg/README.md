# LIVERHCCSEG — EVALUATION-ONLY dataset (independent-cohort HCC segmentation)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

LiverHccSeg (Iyer, Chomicki, Partovi, Graham et al., *Data in Brief* 2023,
TCGA-LIHC-derived) provides real expert liver + HCC tumour segmentation masks on
multiphasic CE-T1w MRI. Here it serves a specific purpose distinct from
`lld-mmri-hcc`: it's genuinely **HCC-only** (same disease as atlas-liver-hcc, unlike
lld-mmri-hcc's broader liver-lesion pool) and same **contrast family** as ATLAS's
training data (CE-T1w) — so it doesn't test cross-CONTRAST robustness the way
lld-mmri-hcc's T2WI/DWI does. What it adds is an independent **cohort/scanner/protocol**
(TCGA-LIHC vs. ATLAS's own Dijon cohort), scored with **real expert** annotations (2
independent board-certified radiologist raters) rather than lld-mmri-hcc's MedSAM2
semi-automated ones — an additional concordant stratum in the pooled significance test,
strengthening the evidence without introducing the cross-disease confound that including
non-HCC lesions would.

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_liverhccseg/` subdirs are intentionally empty (no `4_splits`/preprocessed
  content either).
- The pipeline only **consumes** atlas-liver-hcc-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the atlas-liver-hcc model folders, and `06_evaluate` scores
  the results against this dataset's own tumour ground truth.

## Data

- **17 TCGA-LIHC patients**, 4 native phases each (pre-contrast, arterial,
  portal-venous, delayed CE-T1w) — only **14/17 have tumour annotations** (3 are
  liver-only: `TCGA-BC-4073`, `TCGA-BC-A216`, `TCGA-DD-A4NB`) and are excluded from the
  nnUNet test-input build since only `tumour` is scored here.
- Source: Zenodo record `10.5281/zenodo.7957515` (current version 1.1,
  `zenodo.org/records/8179129`), fully public, no TCIA account/tooling needed
  (self-contained NIfTI + segmentations bundle). License **CC BY 4.0**.
- **Two independent board-certified radiologist raters** annotated each case — this is
  the dataset's own methodological contribution (studying inter-rater agreement), so it
  deliberately provides no single blessed consensus. This pipeline uses **rater1** as
  the active ground truth (arbitrary but consistent and documented); rater2's masks are
  hard-linked into `1_BIDS_liverhccseg/derivatives/` for reference but not read by
  `05_00_build_test_inputs.py`.
- One patient (`TCGA-BC-A10Y`) has 3 separate tumour instances
  (`rater{1,2}_tumor{1,2,3}.nii.gz`) — merged into one binary lesion mask (logical OR)
  during BIDSification, since atlas-liver-hcc's tumour class doesn't distinguish
  instances.
- Subject ids: sequential `sub-liverhccseg000`..`016` (TCGA barcodes aren't BIDS-legal
  as-is — contain hyphens; original id + study date preserved in `participants.tsv` and
  `case_id_map.json`).

## Evaluation specifics

- atlas-liver-hcc models emit 3 labels (0=background, 1=liver, 2=tumour). This
  dataset's tumour GT is stored as **GT label 1**, so evaluation uses
  **`--label_map '{"tumour": [2, 1]}'`** — same mechanism as `lld-mmri-hcc` /
  `msd-spleen`. **`liver` is deliberately NOT scored** here even though this dataset
  has real liver masks too, so every OOD/cross-dataset column in the pooled
  significance test (`atlas-liver-hcc_cross_dataset_t1w_01_results.yaml`) measures the
  same construct (tumour Dice) as lld-mmri-hcc's columns.
- No FOV restriction — a lesion target has no organ-boundary anchor concept.
- Single input channel per item matches atlas-liver-hcc's single-channel T1w model.
- Methods evaluated: the atlas-liver-hcc 6-method suite (`baseline`, `auglab_default`,
  `synthseg_noEM`, `synthseg_EM`, `srcsm`, OURS
  `auglabAug_v26_6_2_train050_val000`/`val100`).
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).

## License note

LiverHccSeg is released under **CC BY 4.0**. Cite Iyer et al., *Data in Brief* (2023),
DOI `10.1016/j.dib.2023.109680`, and the Zenodo record `10.5281/zenodo.7957515`.
