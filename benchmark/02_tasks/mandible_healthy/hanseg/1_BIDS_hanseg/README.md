# HaN-Seg (headneck-hanseg) — test-only

42 head-and-neck radiotherapy patients (Podobnik et al., *Med. Phys.* 2023;
Zenodo 10.5281/zenodo.7442914; **CC BY-NC-ND 4.0**), used here ONLY as the
cross-modality (CBCT → CT) test set for `toothfairy2`-trained models. Nothing is
trained on this dataset.

What is built here is **not** the raw release: every volume is cropped to a
fixed-size, mandible-centred box matched to toothfairy2's training field of view,
and resampled to 0.6 mm isotropic. See
`5_scripts_hanseg/01_prepare/01_01_prepare_ct.py` for the FOV rationale, the
explicit leakage statement, the measured GT-retention numbers, and why the MR arm
is deliberately not built (CT and MR are unregistered; all OAR masks live on the
CT grid only).

> **⚠️ CORRECTION 2026-09-17 — THE LABEL CLAIM BELOW IS WRONG. RESOLVED: re-scored,
> no result changed.**
> HaN-Seg's `Bone_Mandible` **EXCLUDES the teeth**, it does not include them. Verified three ways:
> (a) Brouwer et al. 2015 consensus guidelines, verbatim — "The mandible was defined as the entire
> mandible bone, **without teeth**"; (b) HaN-Seg's own paper (Podobnik et al., Med Phys 2023 §2.2)
> states it followed those guidelines (its ref 10) and declares its deviations — mandible is not
> among them; (c) empirically on our own copy — inside the mask p99 ≈ 1500 HU (cortical bone, no
> enamel population), and 96.6–99.7% of enamel-range (>2000 HU) voxels within the mandible's
> bounding box lie OUTSIDE the mask, as a shell starting at the mask surface.
>
> Consequence: union scoring imposes a systematic false-positive penalty on every method on both
> hanseg arms (`lower_teeth` ≈ 13.1% of the union volume; a perfect union prediction caps at ~93%
> Dice). The penalty scales with how much tooth a method predicts, so it is **not uniform across
> methods and could in principle mask real separation. **It does not — this was re-scored on
> 2026-09-17 and the outcome is a clean negative result.**
>
> Everything was re-evaluated mandible-only on ALL eval sets (in-domain CBCT + both hanseg arms,
> 11 runs x 3 folds, 99 evaluations): **every significant result stays significant and every tie
> stays a tie.** OOD macroΔ vs Ours, union -> mandible-only: auglab_default −0.01 -> −0.17
> (Holm 1.0000 both), synthseg_EM −0.63 -> −0.20 (1.0000 both), baseline +26.02 -> +27.05,
> srcsm +21.37 -> +20.97, synthseg_noEM +28.58 -> +28.59. Dice rises ~1 point across the board;
> ordering unchanged. The ladder also holds: rung 4->5 +0.51 -> +0.54, and the +AugLab cost on
> hanseg_ct −7.10 -> −7.01.
>
> **So the union scoring is wrong on the FACTS and the text below should not be trusted, but no
> reported number needed retracting.** The label-consistent view now lives in parallel roots
> (`02_metrics_mandible_only/`, both datasets) alongside the 3-class and union views — nothing was
> overwritten. Built by `06_10_eval_mandible_only_cbct.sh` + hanseg `06_03_eval_mandible_only.sh`,
> config `toothfairy2_mandible_only_01_results.yaml`.
>
> Mandible-only is the better match but still not exact: it under-covers by the tooth ROOTS
> (ceiling ~97.8% vs the union's ~94.9%), because HaN-Seg is a solid bone envelope with roots
> inside while toothfairy2's `mandible` carves the sockets out. An exact target
> (`mandible ∪ (lower_teeth ∩ slicefill(mandible))`) is constructible if ever wanted; given a
> correction of this size moved nothing, a further ~2% refinement is very unlikely to matter.
> The 3-class task reduction is unaffected (it rests on the independent annotation-gap audit).

Shared class with toothfairy2: `mandible` — which equals toothfairy2's
`mandible` ∪ `lower_teeth` (HaN-Seg's Bone_Mandible includes the lower dentition).
Predictions must be collapsed by `05_predict/05_20_merge_mandible_union.py` before
evaluation.
