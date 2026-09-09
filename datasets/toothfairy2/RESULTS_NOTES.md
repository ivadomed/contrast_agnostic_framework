# ToothFairy2 (CBCT) — task record for the paper

Everything a write-up needs: provenance, why each design choice was made, what the
numbers are, and every caveat that must travel with them. Last updated 2026-09-08.

---

## 1. What this task is, and why it exists

The project's first **non-CT, non-MRI training task**. Every other training set is MRI
(on-harmony, open-ms, brats2024-glioma, chaos, ispy2), with CT appearing only as a
cross-dataset test modality. CBCT is genuinely different acquisition physics — cone-beam
geometry, heavy scatter, no calibrated Hounsfield scale, metal beam-hardening — while
staying volumetric and anatomically framed like CT/MRI (which is why ultrasound was
ruled out: its field of view is not comparable).

Scientifically it exists to populate the **boundary-defined** pole of the paper's
dissociation. Its targets are separated by tissue interfaces (cortical bone against air
and soft tissue, enamel against bone, airway air against mucosa), the opposite of
texture-defined targets like open-ms lesions or brats tumour sub-regions. Before this,
chaos organs was the ONLY boundary-defined ladder; this is a second, independent one on
a new modality.

## 2. Data provenance and licences

| | source | licence | n |
|---|---|---|---|
| training / in-domain | ToothFairy2 challenge (MICCAI 2024), Univ. Modena & Reggio Emilia + Radboud UMC | **CC BY-SA 4.0** | 480 volumes (479 usable) |
| OOD arm 1 + 2 | HaN-Seg (Podobnik et al., *Med. Phys.* 2023; Zenodo 7442914) | **CC BY-NC-ND 4.0** | 42 patients, CT + T1w MR |

ToothFairy2 is redistributable with attribution + share-alike, so it is a valid future
git-annex upload candidate and qualitative figures using it are unambiguously fine.
HaN-Seg is **NC-ND** — fine to evaluate on, but do NOT redistribute derivatives.

Obtained via a complete Hugging Face mirror (`XXCHENZm/ToothFairy2`, 26 GB) since the
official Ditto page gates download behind free registration; the CC BY-SA licence makes
the mirror legitimate.

## 3. Task definition — 3 classes, derived from evidence not preference

**mandible (1), lower_teeth (2), pharynx (3)**, at 0.6 mm isotropic (down from the
release's 0.3 mm — 8x fewer voxels, applied identically to all methods, so it cannot
bias the comparison; it does mean absolute Dice is not comparable to the ToothFairy2
leaderboard, which we never claim).

The reduction from the release's 42 classes was forced by a full per-source-id
histogram over all 480 volumes (`00_utils/00_01_audit_source_labels.py` ->
`source_label_audit.json`). The release ships **two cohorts with different annotation
completeness**:

| | F cohort (63) | P cohort (417) |
|---|---|---|
| Lower Jawbone | 63/63 | 416/417 |
| lower teeth (31-48) | 62/63 | 393/417 |
| Pharynx | 63/63 | 417/417 |
| Upper Jawbone | 61/63 | **124/417** |
| upper teeth (11-28) | 60/63 | **296/417** |
| Maxillary Sinus | 45/63 | **4/417** |

The proof this is an ANNOTATION gap and not anatomy: of P's 293 maxilla-missing cases,
**173 nonetheless have upper TEETH labelled** — you cannot have upper teeth without an
upper jawbone. Training on those classes would have explicitly supervised the network to
call plainly visible anatomy "background" in ~70% of cases.

Consistency was checked on EXTENT, not just presence: mandible and lower_teeth median
volumes agree across cohorts to ~10%. Pharynx looked 2x larger in F, but that is field
of view (F median z-extent 82 mm vs P 51 mm); as a FRACTION of imaged volume the
annotation matches (F 1.63%, P 1.88%).

Also dropped: inferior alveolar canals (thin, frequently no visible cortical boundary —
the ambiguous character this task exists to avoid) and prosthetics (manufactured
material, not tissue; a method that synthesises tissue appearance has no notion of
titanium).

## 4. Splits

479 usable cases (ToothFairy2P_077 excluded — no mandible label at all). 408 train pool
/ 71 sealed test. 3 folds (CLAUDE.md policy: folds 0/1/2 only).
**Stratified on COHORT first, then dentition burden** — the F/P FOV difference is a real
domain shift and F is only 13% of the release, so a plain random draw could leave a fold
with almost no wide-FOV cases. Result: 18 F cases in every validation fold; test set 9 F
/ 62 P.

## 5. Training

1000 epochs, 3 folds, 11 runs = the 6-method suite + 4 extra causal-ladder rungs
(+ the DualVal val100 mirror). Measured on H100: 17-38 s/epoch for the single-validation
methods; **srcsm was only ~1.1x slower here, NOT the ~3x seen on brats/on-harmony** — the
per-dataset measurement CLAUDE.md demands. The real stragglers were the
second-validation-pass trainers (DualVal / ValSynth) at ~50-105 s/epoch.

## 6. Evaluation design — one in-domain arm, two OOD modalities

| arm | image | ground truth | n |
|---|---|---|---|
| in-domain | held-out CBCT | ToothFairy2's own, 3 classes | 71 |
| OOD 1 | HaN-Seg CT | HaN-Seg `Bone_Mandible` | 42 |
| OOD 2 | HaN-Seg **MR-T1 resampled into the CT frame** | **the same** `Bone_Mandible` | 41 |

**Label correspondence is exact, not approximate.** HaN-Seg delineates ONE
`Bone_Mandible` that INCLUDES the lower dentition; that is precisely
`mandible ∪ lower_teeth`. This is why the task was reduced to the mandibular block in
the first place. Predictions are collapsed to that union before scoring
(`05_20_merge_mandible_union.py`).

FOV is matched: HaN-Seg CT is 571 x 571 x 404 mm against a 112 x 104 x 51 mm training
median, so each case is cropped to a fixed-size mandible-centred box (144 x 128 x 80 mm,
near the training p90). Measured: crops 144-153 x 128-133 x 80-91 mm, GT retention
median 100% / min 95.3%. Image and label are cropped together, so excluded voxels are
never scored as false negatives.

### 6.1 The MR arm — how it is built, and why it is defensible

**The MR image is moved; the ground truth is never touched.** MR is rigidly registered
to CT and resampled onto the EXISTING cropped-CT grid, then scored against the SAME
`labelsTs_ct` the CT arm uses. Verified: the label checksum is byte-identical before and
after (`931891e63322f44e`). Consequently the `ct` and `mrt1` columns differ ONLY by
image modality — same GT files, same frame, same FOV crop.

This follows the dataset's documented intended use:
- Dataset paper §2.2: *"each MR image was first registered to the CT image of the same
  patient, and then OARs were annotated in the reference coordinate system of the CT
  image."*
- Challenge report: the released images are deliberately **non-registered**
  ("simulated a real-world clinical scenario"), and of the 5 reporting teams, 3
  registered MR->CT themselves — the winner (eli1) with **rigid SimpleElastix**,
  CHB-QuantIF with **rigid ANTsPy**, Mamaa with a **pure translation**.

The transform is near-trivial, measured not assumed: rotations **0.23-6.74 deg (median
1.03)**, and the y-translation clusters near a constant ~100 mm while z varies per scan
(-597 to +165). That signature is a coordinate-origin convention difference plus DICOM
table position — not patient motion. These are RT-planning scans in an immobilization
mask, which is also why a translation alone sufficed for one challenge team.

**Deviation to disclose:** SimpleElastix is not in the Alliance wheelhouse and pulling
ITK from PyPI risks this project's shared venv, so we used SimpleITK's rigid Mattes-MI
registration — the same rigid+MI method class — with deterministic (REGULAR) sampling
for reproducibility.

**Limitation to report:** residual registration error lives in the IMAGE, so a perfect
model cannot score 100% on the MR arm. This is unavoidable and symmetric (you cannot
escape it by choosing which volume to move), it is the situation every challenge
participant was in, and it does NOT bias the between-method comparison — all methods see
identical inputs and identical GT. Case counts differ slightly (42 CT vs 41 MR;
case_15 fails registration QC reproducibly and is excluded).

### 6.2 A rejected approach, recorded so it is not retried

An earlier MR arm transported the LABELS into MR space. It is built and disabled
(`01_02_prepare_mr.py`). It was rejected because its QC gate was **anti-correlated with
accuracy**: displacing a mask by 12 mm RAISED `tissue_frac` from 0.943 to 0.967, because
sliding the mask off bone onto soft tissue increases the "on tissue" fraction (bone is
dark in T1). It would have passed a centimetre-scale error. `edge_score` (MR gradient on
the mask boundary) did peak sharply at zero displacement, so the registration itself had
support — the gate did not. Two transferable lessons: **a QC metric must be shown to
respond to the error it is meant to catch** (test it with synthetic displacement,
`01_03_validate_mr_registration.py`), and **a zero-variance statistic is a bug signal,
not convergence** (a centroid shift of exactly 0.00 on all 42 cases revealed a
refinement pass that never ran).

### 6.3 Why there is no third modality

No public MRI or PET dataset annotates mandible, teeth, or the pharynx-in-occlusion.
Checked and ruled out: TotalSegmentator MRI (craniofacial classes are **CT-only**; its
56 MR structures are abdominal/whole-body), the NPC MRI set (277 patients, tumour only),
HNTS-MRG 2024 (150 cases, GTVp/GTVn only), the 3D vocal-tract MRI database (53 volumes —
labels the airway INCLUDING the open oral cavity during phonation, a different structure
from pharynx in occlusion), HaN-Seg's own 30 OARs (no pharynx; its airway labels are
glottis/larynx/cricopharyngeus, all below), and the 2025-26 dental releases (panoramic
X-ray or CBCT). TMJ MRI cohorts with condyle segmentations exist (100/140, 618, 840
cases) but none are released. **PET is excluded for physics, not availability**: cortical
bone has near-zero FDG uptake at ~4-5 mm resolution, so every method would score ~0 and
the test would be vacuous rather than hard.

## 7. Results (FINAL — two OOD modalities, CT + MRI)

### 7.0 THE MOST IMPORTANT METHODOLOGICAL FINDING

**Adding the second OOD modality reversed the headline conclusion.** With hanseg CT as
the only OOD arm, OURS did NOT beat the baseline (one-sided p = 1.0000) and the task
read as a null result. With CT **and** MRI, OURS beats the baseline at **p = 1.3e-08**.

The reason is visible in the per-modality numbers: CBCT and CT are both X-ray
attenuation, so the CT arm barely discriminates — the non-contrast-agnostic baseline
still reaches 75.8 Dice there. MRI is the actual test of contrast-agnosticism, and the
baseline **collapses to 2.98 Dice** while every augmentation method reaches 49-57.

Do not report a single-OOD-modality version of this task: it understates the effect to
the point of inverting the conclusion.

### 7.1 Causal-ablation ladder — per modality and pooled

| rung | adds | CT | MRI | pooled Dice | pooled HD95 | ΔDice | ΔHD95 |
|---|---|---|---|---|---|---|---|
| baseline (floor) | — | 75.85 | **2.98** | 39.85 | 30.75 | | |
| +kmeans | K-means clustering | 79.85 | 49.16 | 64.69 | 33.60 | +24.84 | +2.85 |
| +label_remap | label remap | 80.64 | 50.81 | 65.91 | 35.63 | +1.22 | +2.03 |
| +voronoi (noise fill) | Voronoi sub-parcellation | **83.22** | 53.71 | 68.64 | 31.90 | +2.74 | −3.73 |
| **v26_6_2 (real fill)** | **same partition, REAL fill** | 82.86 | **55.11** | **69.15** | **31.66** | **+0.51** | **−0.25** |
| +AugLab (val000) | full AugLab recipe | 75.76 | 55.10 | 65.55 | 35.87 | −3.60 | +4.22 |
| +AugLab (val100) | 100%-synth validation | 75.92 | 55.19 | 65.68 | 35.49 | +0.13 | −0.39 |

**Rung 4->5 (noise fill -> real-intensity fill, partition identical) = +0.51 Dice /
−0.25 HD95 pooled** (CT −0.35, MRI +1.40). Near-zero, which is the prediction for a
BOUNDARY-defined target, against ~+7 Dice on texture-defined ones (open-ms lesions,
brats sub-regions). The conclusion is unchanged by adding the second modality, and is
now measured across two.

### 7.2 Headline table (Dice; `sig. vs ref` = Holm-corrected one-sided "OURS better")

| method | cbct | hanseg_ct | hanseg_mrt1 | all | sig. vs ref |
|---|---|---|---|---|---|
| baseline | **95.4** | 75.8 | 3.0 | 58.1 | **1.3e-08** |
| auglab_default | 94.3 | **75.9** | 55.0 | **75.1** | 0.5976 |
| synthseg_noEM | 78.8 | 48.8 | 24.9 | 50.8 | **7.5e-25** |
| synthseg_EM | 90.9 | 75.5 | **56.7** | 74.3 | 0.0664 |
| srcsm | 93.1 | 63.2 | 24.9 | 60.4 | **1.1e-14** |
| **OURS (train050_val000)** | 94.1 | 75.8 | 55.1 | 75.0 | — |

OURS decisively beats baseline (1.3e-08), synthseg_noEM (7.5e-25) and srcsm (1.1e-14);
ties auglab_default (p=0.60); is marginally ahead of synthseg_EM (p=0.066, and
synthseg_EM is actually best on the MRI column at 56.7).

## 8. FINDINGS THAT MUST NOT BE BURIED

**(a) OURS ties auglab_default and only marginally leads synthseg_EM.** The clear wins
are over baseline, srcsm and synthseg_noEM. Report the ties as ties.

**(b) The full AugLab recipe COSTS 7.10 Dice on the CT arm** (rung 5 82.86 -> rung 6
75.76) while being neutral on MRI (55.11 -> 55.10). So the penalty is modality-specific,
which the pooled column hides (−3.60). Unexplained. Leading hypothesis, NOT verified:
rungs 2-5 use spatialDA-only configs while OURS adds full `default01-23` intensity
augmentation. Needs an explicit ablation; on the CT arm the best configuration remains
an intermediate rung.

**(c) HD95 ranks the baseline BEST overall (20.9), which is a metric artifact.** Its
MRI Dice is 3.0 — it predicts almost nothing, and HD95 on near-empty predictions is not
comparable to HD95 on real ones. Do not quote the HD95 `all` column without the Dice
column beside it.

**(d) Label-consistency caveat.** The `cbct` column is a 3-class macro while both hanseg
columns are a single class, so `all` mixes label sets and the in-domain->OOD drop
conflates modality with label set. `06_08_eval_mandible_union.py` produces the
label-consistent view; the LADDER is already consistent (OOD only).

**(e) The FOV crop is GT-centred.** Position leaks (extent does not — fixed-size box),
applied identically to all methods, so the comparison is unbiased but absolute Dice is
inflated vs a real localize-then-segment pipeline.

**(f) Case counts differ slightly between OOD arms**: 42 CT vs 41 MRI (case_15 fails
registration QC reproducibly and is excluded).

## 9. Reproducing every table

```
D=datasets/toothfairy2/5_scripts_toothfairy2/06_evaluate
P=/scratch/paulh/tf2_packs; S=20260908_013025
bash $D/06_04_write_configs.sh $P/suiteA_$S $P/suiteB_$S $P/ladder_$S
bash $D/06_06_cross_dataset_summary.sh       # headline (in-domain + hanseg)
bash $D/06_07_combined_modality_summary.sh   # meta-heatmap feed
bash $D/06_02_aggregate_from_config.sh       # in-domain, 3 classes
bash $D/06_03_significance_from_config.sh $D/configs/toothfairy2_cross_dataset_01_results.yaml
bash $D/06_05_ladder_summary.sh $P/suiteA_$S $P/suiteB_$S $P/ladder_$S
```
All are thin wrappers over `datasets/00_commun_scripts/00_03_evaluate/` — no ad-hoc
tables or p-values. Configs are ${PROJECT_ROOT}-relative (never ${METRICS_ROOT}, which
has previously produced a directory literally named after the unexpanded variable).
