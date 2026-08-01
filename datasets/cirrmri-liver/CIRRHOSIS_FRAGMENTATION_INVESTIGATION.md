# CIRRMRI-LIVER: prediction fragmentation investigation — EXCLUDED, unresolved

**Bottom line: excluded from the cross-dataset roll-up tables (both
`cross_dataset_t1in_01_results.yaml` and `cross_dataset_t2spir_01_results.yaml`,
2026-08-01). Every plausible pipeline mechanism was checked and ruled out with
direct evidence. The leading remaining explanation (cirrhosis-driven anatomical
distortion, e.g. splenomegaly) is plausible and consistent with the dataset's
own diagnosis metadata, but not independently confirmed — this dataset's own
GT is liver-only, so there is no spleen/kidney label to check the hypothesis
against directly.**

## Trigger

While pulling prediction paths for visualization (per user request, after the
[[project_fov_hard_crop_variant]] work), user reported: *"the models seem to
split the spleen into chunks and label parts of it as liver, spleen and
kidney, and consistently miss the kidneys."*

## Symptom, verified directly on real voxel data

Checked `CR177`, `CR174`, `CR094` (largest OURS-vs-best-other gaps on the
fov_crop t1in branch). Both **OURS and srcsm** — not just one method — show
severe fragmentation on the SAME cases:

| case | method | liver | R.kidney | L.kidney | spleen |
|---|---|---|---|---|---|
| CR177 | OURS | 14 components, 63% largest | 6 comp, 48% | 7 comp, 50% | 8 comp, 58% |
| CR177 | srcsm | 30 components, 71% largest | 36 comp, 56% | 39 comp, 63% | 24 comp, 41% |
| CR174 | OURS | 5 comp, 98% | 8 comp, 49% | 6 comp, 44% | 4 comp, 76% |
| CR094 | OURS | 16 comp, 92% | 7 comp, 90% | 12 comp, 97% | 13 comp, 43% |

A healthy prediction is 1 connected component per organ. This is present in
the **original, full-context, pre-crop** prediction (recorded masked-eval Dice
for CR177/OURS was already 0.187 before any of this session's crop work) —
i.e. it predates and is independent of the FOV hard-crop investigation.

## Ruled out, each with a direct test (not assumption)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | FOV hard-crop pipeline | **No** | Fragmentation present in archived pre-crop predictions, identical pattern. |
| 2 | L-R orientation flip | **No** | GT liver centroid fraction along axis0 nearly identical between chaos (0.28-0.37) and cirrmri-liver (0.27-0.36). |
| 3 | Intensity/normalization outliers | **No** | Intensity ranges (0-700 to 0-1500) comparable to chaos's own training range (0-1136); z-score normalization is per-image, not global training stats (confirmed via nnU-Net source, `ZScoreNormalization.run`). |
| 4 | Trainer-class inference-code difference (OURS uses `nnUNetTrainerCHAOSAugLabV26_6_2`/`ValSynth`, others use `nnUNetTrainerCHAOSAugLabDefault`) | **No** | Read the source: both OURS trainers only override training-time methods (`validation_step`, `get_validation_transforms`, wandb logging) — inherit architecture/inference code unchanged. Doesn't explain why srcsm (same trainer as the healthy methods) *also* fragments. |
| 5 | Patch/tile-count mismatch (cropped volumes far larger than chaos's training images) | **No** | Initial calculation wrongly used pre-resampling shape. After correctly accounting for nnU-Net's resample-to-target-spacing step, tile counts are comparable across chaos/amos/cirrmri/msd-spleen/sliver07 (~1-8 tiles). |
| 6 | Config override (real inference uses different plans.json than assumed) | **No** | md5-compared the plans.json actually shipped in the real prediction output dir against the reference copy — byte-identical. |
| 7 | Resampling/interpolation artifact (`order_z=0` nearest-neighbor z-downsampling introducing aliasing) | **No** | Manually pre-resampled CR177/CR174 to chaos's exact target spacing using nnU-Net's own `resample_data_or_seg_to_shape` (order=3/order_z=0, identical call), so nnU-Net's on-the-fly resampling becomes a no-op. Fragmentation persisted essentially unchanged (OURS/CR177: still 17 components, 79% largest-fraction). |

## Leading explanation (unconfirmed)

CIRRMRI-LIVER is a **diagnosed cirrhosis cohort** — confirmed via
`0_raw_cirrmri-liver/Metadata/T1_age_gender_evaluation.csv` +
`Metadata/labels.txt` (Radiological Evaluation: 1=Mild, 2=Moderate, 3=Severe).
CR177=Moderate, CR174=Mild, CR094=Mild — not a clean severity gradient across
these 3 cases, but that doesn't refute the hypothesis: splenomegaly (spleen
enlarging toward/past liver size) is a downstream consequence of portal
hypertension in cirrhosis and doesn't necessarily track a radiologist's
liver-nodularity grade 1:1.

If true, this is a genuine and important limitation shared by **every**
method here: all of chaos's augmentation strategies (SynthSeg-style,
AugLab default, srcsm, OURS's domain randomization) randomize image
**appearance/contrast**, not anatomical **morphology** — none simulate
organ-size or organ-ratio changes from disease. A model that's never seen a
spleen approaching liver size has no reason to segment one correctly,
regardless of how its training contrast was randomized. This would explain
why the failure is shared across methods (not method-specific) and why it's
concentrated in specific cases rather than uniform across the dataset.

**Not independently confirmed** because CIRRMRI-LIVER's own GT is liver-only —
there is no spleen or kidney label to directly measure against and verify
enlarged-spleen extent. Confirming this would need either manual
radiological read of the flagged cases, or a spleen segmentation from an
independent (non-chaos-trained) source.

## Practical outcome

Excluded from `cross_dataset_t1in_01_results.yaml` /
`cross_dataset_t2spir_01_results.yaml` (column + source block removed,
2026-08-01) — same treatment as `kidney-t2w`
([[project_kidney_t2w_onboarding]], excluded for an unrelated
acquisition-geometry reason) and `trusted` (excluded for a center-bias
artifact). CIRRMRI-LIVER's own per-dataset results
(`8_results_cirrmri-liver/`) are left as-is for reference but should be read
with this caveat.

## Reproduction / artifacts

- Manual resampling test (rules out #7): saved at
  `.scratch_analysis/manual_resample_test/` — `{CR177,CR174}_resampled_0000.nii.gz`
  (+ `_gt.nii.gz`), predictions in `pred_srcsm/` and `pred_ours/`.
- Fragmentation connectivity checks: ad-hoc, not saved as a script (this was
  interactive investigation). To reproduce: load a case's prediction NIfTI,
  `scipy.ndimage.label` per chaos class (1=liver, 2=R.kidney, 3=L.kidney,
  4=spleen), count components and largest-component fraction.
