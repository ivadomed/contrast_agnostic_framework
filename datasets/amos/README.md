# AMOS — Abdominal Multi-Organ Segmentation

> **Current status: inference-only** (chaos-trained models applied to AMOS).
> Training pipeline on AMOS itself is **pending** — see "Roadmap" below.

AMOS22 (NeurIPS 2022) — 500 CT + 100 MRI scans from multi-center, multi-vendor
clinical scenarios, voxel-level annotations of **15 abdominal organs**.

## Data

| Split | CT | MRI | Source IDs |
|---|---|---|---|
| Labeled (train+val) | 499 | 101 | CT: 0001–0499, MRI: 0500–0599 |
| Unlabeled (not downloaded) | 2000+ | 1200 | skipped — not needed for current use |

Raw NIfTI in `0_raw_amos/amos22/` (nnUNet-style layout: `imagesTr/`, `labelsTr/`,
`imagesVa/`, `labelsVa/`, `imagesTs/`, `dataset.json`).

**15 labels:** background(0), spleen(1), right_kidney(2), left_kidney(3),
gallbladder(4), esophagus(5), liver(6), stomach(7), aorta(8), inferior_vena_cava(9),
pancreas(10), right_adrenal_gland(11), left_adrenal_gland(12), duodenum(13),
right_bladder(14), prostate_uterus(15).

## Roadmap

1. **Now — inference:** apply chaos-trained models (MR T1-in) to AMOS CT/MRI as
   additional cross-dataset generalization probe. Predictions under
   `8_results_amos/01_predictions/chaos_models/`.
2. **Later — training:** train domain-randomization models (v26_6_2, synthseg, etc.)
   directly on AMOS, then evaluate them on `chaos` and `sliver07`.
   This requires the full `01_create_splits → 02_nnunet → 03_preprocess → 04_train`
   pipeline (currently empty; the structure is ready).

## Citation

```bibtex
@inproceedings{NEURIPS2022_ee604e1b,
  author    = {Ji, Yuanfeng and Bai, Haotian and GE, Chongjian and Yang, Jie and
               Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhanng, Lingyan and
               Ma, Wanling and Wan, Xiang and Luo, Ping},
  title     = {AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for
               Versatile Medical Image Segmentation},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {36722--36732},
  year      = {2022}
}
```
