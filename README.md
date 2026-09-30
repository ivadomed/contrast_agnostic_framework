# PALETTE — Texture-Preserving Contrast Augmentation for MRI Segmentation

A single segmentation model, trained on **one** MRI contrast, that generalizes to contrasts it has
never seen — T1w, T2w, FLAIR, GRE, DWI, EPI, CT — without retraining, without per-contrast tuning,
and without a target domain in mind at all.

## Why

MRI segmentation models are notoriously brittle to contrast. The same anatomy, imaged with a
different sequence, can look unrecognizable to a model that was only ever shown one appearance —
even though the shapes, boundaries and texture underneath are identical. The usual fixes are either
narrow (harmonize to one target contrast, retrain per site) or expensive (collect labels in every
contrast you care about). We take a third path: **domain randomization**. Instead of picking a
target appearance, we synthesize training images that deliberately explore far beyond any single
real scanner or contrast — so the model never gets the chance to overfit to one appearance in the
first place.

## What we do

One real T1w scan, and four real (unretouched) outputs of the PALETTE transform applied to it —
same anatomy and texture throughout, only contrast changes each time:

![One real T1w input and four real PALETTE-augmented outputs of it — anatomy and texture are preserved, only contrast is randomized](paper/cvpr_format_latex/figures/readme_palette_examples.png)

**PALETTE** (*Partition-based Affine Local-intEnsity Transform for Texture-prEserving
augmentation*) is a label-free, on-the-fly contrast augmentation:

1. **K-means** on foreground intensities → a handful of intensity classes, no atlas or labels needed.
2. **Voronoi sub-parcellation** of each class → finer, still label-free spatial regions.
3. **Signed affine intensity remap** per sub-region (`y = μ + α(x − mean)`, α drawn from both signs,
   so contrast can locally invert as well as rescale).
4. **Per-label decoupling** using the ground-truth segmentation, for the labeled regions that have one.

The key property: PALETTE remaps *intensities* of the real volume — it never redraws anatomy from
noise. Texture, gradients, partial-volume detail and unlabeled structures all survive; only contrast
changes. That distinguishes it from label-generative approaches (e.g. SynthSeg-style GMM synthesis),
which regenerate texture from noise per label and destroy exactly the fine structure we're trying to
preserve. We test this directly with a causal ablation ladder (same spatial partition, only the fill
swapped from noise to real intensities) — see the paper for the full mechanism story.

One config, tuned once, is reused unchanged across every dataset and contrast below — no
per-dataset retuning.

## Results snapshot

Benchmarked across **8 anatomical tasks** (brain tumor, brain MS, healthy brain, breast cancer,
abdominal organs, mandible, pelvis, spine) spanning MRI and CT, each trained on one contrast and
evaluated cross-contrast against 5 baselines (nnU-Net default augmentation, two SynthSeg-style
variants, an existing GPU augmentation library, and a semantically-random-convolution baseline):

| method | overall Dice ↑ | overall HD95 (mm) ↓ | sig. vs Ours |
|---|---|---|---|
| baseline (no augmentation) | 35.7 | 69.7 | p = 8.7e-98 |
| synthseg_noEM | 33.8 | 91.5 | p = 3.8e-100 |
| synthseg_EM | 60.3 | 35.0 | p = 3.9e-47 |
| auglab_default | 62.0 | 32.6 | p = 3.8e-09 |
| srcsm | 54.6 | 36.5 | p = 3.2e-60 |
| **PALETTE (Ours)** | **63.0** | **31.6** | — |

Every comparison is Holm-corrected, patient-level, and computed on held-out contrasts the model
never trained on. Full per-task breakdowns live in `benchmark/01_commun_results/`.

## Repository layout

```
src/                    PALETTE method source (the contrast transform + model) — no pipeline glue
benchmark/
  00_commun_scripts/     shared train/predict/evaluate/aggregate/significance drivers (canonical)
  01_commun_results/     cross-dataset comparison tables, incl. the headline task-level heatmap
  02_tasks/<task>/<dataset>/   one folder per benchmark dataset, grouped by anatomy/pathology
  03_archive/            excluded or superseded datasets, kept for history
scripts/                 cluster job submission (run_job), venv setup, cross-cutting tooling
sub-workspaces/          AugLab (external GPU augmentation library this method builds on)
paper/                   CVPR paper source
```

Every dataset under `benchmark/02_tasks/` follows the same structure — raw data, BIDS, nnU-Net
conversion, splits, pipeline scripts, checkpoints, results, tests — so all datasets train, predict,
evaluate and aggregate identically. This is deliberate: it's what makes the numbers above directly
comparable across 8 completely different anatomical structures. See `CLAUDE.md` for the full
convention if you're adding a new dataset or method.

## Installation

Requires Python 3.11 and an nnU-Net v2 training environment.

```bash
module load python/3.11        # or your own Python 3.11 install
python3 -m venv .venv
source .venv/bin/activate
```

The full, tested dependency recipe (including several packages that need pinning or pre-downloading
to avoid silent version backtracking) lives in `scripts/job_runner/_setup_venv_oneshot.sh` — run it
inside a batch job on an HPC cluster, or adapt it for a local/interactive install. It also registers
this project's nnU-Net trainer classes and pulls in AugLab:

```bash
git submodule update --init sub-workspaces/auglab_workspace/AugLab   # or clone it there directly
pip install -e sub-workspaces/auglab_workspace/AugLab
```

## Running an experiment

Every train / predict / evaluate / aggregate step goes through the shared drivers in
`benchmark/00_commun_scripts/` — never call `nnUNetv2_train`/`nnUNetv2_predict` directly. A new
dataset gets scaffolded, not hand-written:

```bash
.venv/bin/python benchmark/create_dataset_structure.py <dataset_name>
```

To train the full 6-method suite (baseline, two SynthSeg variants, AugLab default, SRCSM, PALETTE)
on one modality of an existing dataset:

```bash
bash benchmark/02_tasks/<task>/<dataset>/5_scripts_<dataset>/04_train/04_XX_run_all_<modality>.sh
```

then predict, evaluate, and aggregate the same way — one config-driven command per step:

```bash
bash .../05_predict/05_01_predict_common.sh <run_id>
bash .../06_evaluate/06_01_evaluate_run.sh <run_id>
bash .../06_evaluate/06_02_aggregate_from_config.sh configs/<dataset>_<modality>_01_results.yaml
```

Before trusting any change to shared code, run the structure validator (it checks both the 9-subdir
layout and the canonical script-numbering convention across every dataset):

```bash
.venv/bin/python benchmark/validate_standard_dataset_structure.py
```

**This project runs on shared HPC clusters (Vulcan / TamIA / Killarney via the Digital Research
Alliance of Canada).** All heavy compute goes through Slurm, never a login node — see `CLAUDE.md`
for cluster-specific submission, storage and GPU policy if you're running this on one of them.

## Citation

Paper source and build instructions are under `paper/`. Citation details will be added once
published.
