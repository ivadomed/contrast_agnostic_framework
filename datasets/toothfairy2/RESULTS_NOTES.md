# ToothFairy2 (CBCT) — results notes, 2026-09-08

Complete: 11 runs x 3 folds trained; predict + evaluate done on both axes; 33/33 metric
files each, exact row counts (213 = 71 held-out cases x 3 labels in-domain; 42 = 42
cases x mandible on hanseg CT). The disabled propagated-GT MR arm never leaked
(`mrt1_metrics.csv` count stayed 0).

Regenerate every table below with (from repo root, Vulcan):
```
D=datasets/toothfairy2/5_scripts_toothfairy2/06_evaluate; P=/scratch/paulh/tf2_packs; S=20260908_013025
bash $D/06_04_write_configs.sh $P/suiteA_$S $P/suiteB_$S $P/ladder_$S
bash $D/06_06_cross_dataset_summary.sh      # headline
bash $D/06_07_combined_modality_summary.sh  # meta-heatmap feed
bash $D/06_02_aggregate_from_config.sh      # in-domain
bash $D/06_03_significance_from_config.sh $D/configs/toothfairy2_cross_dataset_01_results.yaml
bash $D/06_05_ladder_summary.sh $P/suiteA_$S $P/suiteB_$S $P/ladder_$S
```

## 1. THE HEADLINE: the ladder's rung 4->5 step is near-zero, as predicted

Cross-modality (hanseg CT) OOD, causal-ablation ladder:

| rung | adds | OOD Dice | OOD HD95 | ΔDice | ΔHD95 |
|---|---|---|---|---|---|
| baseline (floor) | — | 75.85 | 25.25 | | |
| +kmeans | K-means intensity clustering | 79.85 | 30.74 | +4.00 | +5.49 |
| +label_remap | label remap | 80.64 | 30.99 | +0.79 | +0.24 |
| +voronoi (noise fill) | Voronoi sub-parcellation | 83.22 | 24.22 | +2.57 | −6.77 |
| **v26_6_2 (real fill)** | **same partition, REAL-intensity fill** | **82.86** | **23.93** | **−0.35** | **−0.29** |
| +AugLab (val000) | full AugLab recipe on top | 75.76 | 31.85 | −7.10 | +7.92 |
| +AugLab (val100) | 100%-synth validation | 75.92 | 31.43 | +0.16 | −0.42 |

**Rung 4->5 = −0.35 Dice / −0.29 HD95.** That step swaps noise fill for real-intensity
fill with the partition otherwise identical, so it is the one-variable test of whether
texture preservation causally drives Dice. Near-zero here is exactly the paper's
prediction for a BOUNDARY-defined target, against ~+7 Dice on texture-defined ones
(open-ms lesions, brats sub-regions). This is a SECOND independent boundary-defined
ladder alongside chaos organs, on a genuinely new modality — which is what this dataset
was onboarded to provide.

## 2. Headline table (in-domain CBCT + cross-modality hanseg CT)

Dice, `sig. vs ref` = Holm-corrected one-sided "OURS better" macroΔ p:

| method | cbct | hanseg_ct | all | sig. vs ref |
|---|---|---|---|---|
| baseline | **95.4** | 75.8 | **85.6** | 1.0000 |
| auglab_default | 94.3 | **75.9** | 85.1 | 1.0000 |
| synthseg_noEM | 78.8 | 48.8 | 63.8 | 6.8e-19 |
| synthseg_EM | 90.9 | 75.5 | 83.2 | 4.1e-06 |
| srcsm | 93.1 | 63.2 | 78.2 | 4.0e-09 |
| **OURS (train050_val000)** | 94.1 | 75.8 | 84.9 | — |

## 3. TWO FINDINGS THAT MUST NOT BE BURIED

**(a) OURS does not beat baseline or auglab_default on this task.** 84.9 vs 85.6 / 85.1
on the `all` column, p = 1.0000 one-sided both ways. It does significantly beat
synthseg_noEM, synthseg_EM and srcsm. A null result against the no-synthesis references
is CONSISTENT with the thesis on a boundary-defined target — synthesis is not supposed
to help where the target is an anatomical interface — but it is a null result and must
be reported as one, not framed as a win.

**(b) The intermediate ladder rungs BEAT both the floor and OURS, and this is
unexplained.** Rungs 3-5 reach 80.6 / 83.2 / 82.9 OOD Dice against a 75.9 baseline floor
and 75.8 for OURS. Adding the full AugLab recipe on top of v26_6_2 (rung 5->6) COSTS
7.10 OOD Dice and +7.9 mm HD95.

Leading hypothesis, NOT verified: the ladder rungs 2-5 use spatialDA-only configs
(`*_spatialDA_train050.json`) while OURS adds the full `default01-23` intensity
augmentation on top. Heavy intensity augmentation may be actively harmful on a
boundary-defined bone task evaluated under a large modality+FOV shift. Worth an
explicit ablation before this task is written up, because as it stands the best
cross-modality configuration on this dataset is an intermediate rung, not the method.

## 4. Caveats attached to the numbers

- hanseg CT is FOV-matched by a mandible-centred crop whose box is GT-CENTRED (position
  leaks, extent does not). Applied identically to every method so the comparison is
  unbiased, but absolute Dice is inflated versus a real localize-then-segment pipeline.
- hanseg scores the mandible ONLY, as the union mandible + lower_teeth (HaN-Seg's
  Bone_Mandible includes the lower dentition). In-domain scores all 3 classes.
- The MR arm is built but DISABLED (propagated GT; its QC gate was anti-correlated with
  accuracy). See `datasets/hanseg/5_scripts_hanseg/01_prepare/01_02_prepare_mr.py`.
