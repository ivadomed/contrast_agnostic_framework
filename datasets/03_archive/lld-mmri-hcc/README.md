# LLD-MMRI-HCC — EVALUATION-ONLY dataset (cross-contrast HCC lesion segmentation)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

LLD-MMRI-HCC is the **hepatocellular-carcinoma subset of LLD-MMRI** (Lou et al. 2025,
*Neural Networks*; MedSAM2-annotated release, Ma/Yang et al. 2025). Here it serves a
single purpose: **atlas-liver-hcc trains on a single modality (CE-T1w), so it cannot
answer the cross-contrast robustness question on its own** — this dataset supplies
**T2WI and DWI** (both genuinely different contrast mechanisms from CE-T1w) with real
HCC lesion masks, letting atlas-liver-hcc-trained models be tested out-of-modality on
the same disease (HCC).

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_lld-mmri-hcc/` subdirs are intentionally empty (no `4_splits`/preprocessed
  content either).
- The pipeline only **consumes** atlas-liver-hcc-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the atlas-liver-hcc model folders, and `06_evaluate` scores
  the results against this dataset's own lesion ground truth.

## Data

- **157 HCC patients** (category `Hepatocellular_carcinoma` = class 6, filtered from
  the full 498-patient / 7-lesion-type LLD-MMRI dataset — see
  `0_raw_lld-mmri-hcc/hcc_patient_ids.txt` and `LLD_MMRI_Annotation_full.json`), each
  with **T2WI + DWI** phases (of the original 8 phases per lesion — the other 6 are
  non-contrast T1 in/out-phase and 4 CE-T1w timings, same sequence family as
  atlas-liver-hcc, so not useful for a cross-contrast test and not downloaded).
- Source: `wanglab/LLD-MMRI-MedSAM2` on HuggingFace (public, no gate,
  `snapshot_download` with `allow_patterns` targeted to the 157 HCC ids' T2WI/DWI
  files only — 652 MB, not the full ~6 GB/7-class repo).
- Segmentation is **lesion-only** (binary `{0,1}`) — annotated via MedSAM2 in a
  human-in-the-loop pipeline, **not fully expert-drawn from scratch** like
  atlas-liver-hcc's masks. Verified organic (non-box) shape on a sample case before
  onboarding (smooth per-slice area progression, not a rasterized bounding box), but
  this quality caveat should be kept in mind when interpreting results — a weaker
  cross-contrast Dice here could partly reflect annotation noise, not just domain gap.
- No liver-organ mask exists in this dataset — only the `tumour` label is ever scored,
  never `liver`.
- Subject ids: sequential `sub-lldhcc000`..`sub-lldhcc156` (source patient ids mix
  hyphenated/non-hyphenated formats — `MR-400851` vs `MR102385` — not BIDS-legal as-is;
  original id + lesion-instance-suffix preserved in `participants.tsv` and
  `case_id_map.json` for traceability).

## Evaluation specifics

- atlas-liver-hcc models emit 3 labels (0=background, 1=liver, 2=tumour). This
  dataset's GT is lesion-only, stored as **GT label 1**. Because atlas's tumour id (2)
  and this dataset's GT id (1) don't match numerically, evaluation uses the shared
  evaluator's **`--label_map '{"tumour": [2, 1]}'`** cross-label-space mode — the same
  mechanism `msd-spleen` uses for chaos's spleen id (4) vs its own GT id (1).
- No FOV restriction (unlike the chaos-family cross-eval sets) — a lesion target has
  no organ-boundary anchor concept.
- Single input channel per item (T2WI or DWI) matches atlas-liver-hcc's single-channel
  T1w model. nnU-Net applies atlas's T1w normalization statistics to genuinely
  different-contrast input — intentional; this is exactly what the domain-randomization
  methods are built to absorb.
- Methods evaluated: the atlas-liver-hcc 6-method suite (`baseline`, `auglab_default`,
  `synthseg_noEM`, `synthseg_EM`, `srcsm`, OURS
  `auglabAug_v26_6_2_train050_val000`/`val100`) — see `05_predict/05_0[2-8]_predict_atlas_*.sh`.
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).

## License note

LLD-MMRI is released under **CC BY-NC 4.0** (noncommercial research use only). Cite
Lou et al., *Neural Networks* (2025) for LLD-MMRI and Ma, Yang et al. (arXiv:2504.03600)
for the MedSAM2 segmentation annotation.
