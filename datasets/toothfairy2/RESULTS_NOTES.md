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

## 7. Results

### 7.1 Causal-ablation ladder (the reason this dataset exists)

Cross-modality OOD. ⚠️ Numbers below are the CT-arm-only version; they are being
recomputed now that the MR arm exists and the OOD pool spans two modalities.

| rung | adds | OOD Dice | OOD HD95 | ΔDice | ΔHD95 |
|---|---|---|---|---|---|
| baseline (floor) | — | 75.85 | 25.25 | | |
| +kmeans | K-means intensity clustering | 79.85 | 30.74 | +4.00 | +5.49 |
| +label_remap | label remap | 80.64 | 30.99 | +0.79 | +0.24 |
| +voronoi (noise fill) | Voronoi sub-parcellation | 83.22 | 24.22 | +2.57 | −6.77 |
| **v26_6_2 (real fill)** | **same partition, REAL fill** | **82.86** | **23.93** | **−0.35** | **−0.29** |
| +AugLab (val000) | full AugLab recipe | 75.76 | 31.85 | −7.10 | +7.92 |
| +AugLab (val100) | 100%-synth validation | 75.92 | 31.43 | +0.16 | −0.42 |

**Rung 4->5 = −0.35 Dice / −0.29 HD95.** That step swaps noise fill for real-intensity
fill with the partition otherwise identical — the one-variable test of whether texture
preservation causally drives Dice. Near-zero is exactly the prediction for a
BOUNDARY-defined target, against ~+7 Dice on texture-defined ones.

### 7.2 Headline table (Dice)

| method | cbct (in-domain) | hanseg_ct | all | sig. vs ref |
|---|---|---|---|---|
| baseline | **95.4** | 75.8 | **85.6** | 1.0000 |
| auglab_default | 94.3 | **75.9** | 85.1 | 1.0000 |
| synthseg_noEM | 78.8 | 48.8 | 63.8 | 6.8e-19 |
| synthseg_EM | 90.9 | 75.5 | 83.2 | 4.1e-06 |
| srcsm | 93.1 | 63.2 | 78.2 | 4.0e-09 |
| **OURS (train050_val000)** | 94.1 | 75.8 | 84.9 | — |

## 8. FINDINGS THAT MUST NOT BE BURIED

**(a) OURS does not beat baseline or auglab_default here.** 84.9 vs 85.6 / 85.1, with a
one-sided "OURS better" p of 1.0000 against both. It DOES significantly beat
synthseg_noEM, synthseg_EM and srcsm. A null result against the no-synthesis references
is CONSISTENT with the thesis on a boundary-defined target — synthesis is not supposed to
help where the target is an anatomical interface — but it is a null result and must be
written as one, not framed as a win.

**(b) The intermediate ladder rungs BEAT both the floor and OURS, and this is
unexplained.** Rungs 3-5 reach 80.6 / 83.2 / 82.9 against a 75.9 floor and 75.8 for
OURS; adding the full AugLab recipe on top of v26_6_2 costs **7.10 OOD Dice** and
+7.9 mm HD95. Leading hypothesis, NOT verified: rungs 2-5 use spatialDA-only configs
while OURS adds the full `default01-23` intensity augmentation, which may be actively
harmful on a boundary-defined bone task under a large modality+FOV shift. Needs an
explicit ablation. As it stands **the best cross-modality configuration on this dataset
is an intermediate rung, not the method.**

**(c) A label-consistency caveat in the headline table.** The `cbct` column is a 3-class
macro while `hanseg_ct` is a single class, so `all` averages incommensurable quantities
and the in-domain->OOD drop conflates "harder modality" with "different label set".
`06_08_eval_mandible_union.py` produces a label-consistent view (in-domain scored on the
same mandible union) — use that for the cross-modality claim, and the 3-class table only
to characterise the full task. The LADDER is already label-consistent (its OOD values
come solely from hanseg).

**(d) The FOV crop is GT-centred.** Position leaks (extent does not — the box is fixed
size). Applied identically to every method so the comparison is unbiased, but absolute
Dice is inflated relative to a real localize-then-segment pipeline and must never be
presented as clinical performance.

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
