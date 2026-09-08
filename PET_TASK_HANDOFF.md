# Handoff: new PET task (7th training task)

Written 2026-09-08 at the end of the ToothFairy2/HaN-Seg session. Start a fresh
session with this file; it exists so that session does not have to rediscover any of
the below.

## HARD REQUIREMENT: >= 2 modalities available FOR TESTING

Stated by the user 2026-09-08, and clarified: what is required is at least two
modalities **to test on**, so that cross-modality generalization can be measured.
Training on only ONE of them is acceptable — that is exactly the shape ToothFairy2
has (trains CBCT, tests CBCT in-domain + hanseg CT cross-modality). Two TRAINING arms
are a bonus, not the requirement.

What this rules in and out:
- a single-modality dataset with no compatible second-modality test set is OUT — it
  cannot produce the headline result at all;
- a dataset with two modalities where only one is trained is IN.

Two further conditions, both learned the hard way and both easy to violate here:
- the test modalities should be the **SAME PATIENTS** where possible (as chaos
  t1in/t2spir, brats t1n/t2w, ispy2 t1wce/t2w are). Two separate cohorts confounds
  contrast with disease/site, which is what made the atlas-liver-hcc ladder flip sign.
  An external same-label test set is acceptable (chaos -> amos/sliver07; toothfairy2 ->
  hanseg) but the label definitions must match exactly;
- the ground truth must be **genuinely visible in BOTH modalities** — including in the
  one used only for TESTING. A label defined by one modality's physics and evaluated
  on another reproduces, one level down, the exact reason PET was rejected as a
  ToothFairy2 arm: every method scores ~0 and the test measures nothing.

## Why this task

ToothFairy2 (CBCT, onboarded this session) added a new modality on the
**boundary-defined** side of the paper's dissociation table. A PET task is worth
doing because it is:

1. **A genuinely new modality.** The project currently trains on MRI (on-harmony,
   open-ms, brats2024-glioma, chaos, ispy2) and CBCT (toothfairy2), with CT only ever
   a cross-dataset *test* modality.
2. **Texture-defined, not boundary-defined.** PET lesions are metabolic hot spots with
   no anatomical interface — the opposite pole from ToothFairy2's mandible/teeth. The
   ladder's rung 4->5 step (noise fill -> real fill) is predicted LARGE here and is
   near-zero on boundary-defined targets. That dissociation is the paper's central
   causal claim, so a second independent texture-defined task strengthens exactly the
   axis that matters.

## Why it is a NEW TASK and not an arm on ToothFairy2

Do not try to bolt PET onto toothfairy2. Investigated and ruled out this session:
**the mandible is physically invisible in FDG PET** (cortical bone has near-zero
uptake; PET resolution ~4-5 mm). Labels could be obtained legitimately — TCIA head/neck
PET-CT collections ship RTSTRUCT with clinician mandible contours, and PET/CT is
HARDWARE-registered at acquisition, so unlike the HaN-Seg MR arm no self-made
registration would be involved — but every method would score ~0 for reasons unrelated
to domain randomization. The test would be vacuous, not hard.

Same conclusion for MRI, for a different reason: no public MRI dataset annotates
mandible or teeth (TotalSegmentator MRI's craniofacial classes are CT-ONLY; dental-MRI
cohorts are study-internal), and the one open manually-annotated airway MRI database
(53 vocal-tract volumes, 10 French speakers) labels the airway INCLUDING the open oral
cavity during phonation — a different structure from `pharynx` in occlusion. NPC MRI
(277 patients) and HNTS-MRG 2024 (150 cases) are both tumour-only.

## Candidate datasets — ranked against the 2-modality requirement

**HECKTOR 2022 is the recommendation.** (This reverses an earlier draft of this file
that preferred AutoPET; the 2-modality requirement is what flips it.)

| | HECKTOR 2022 | AutoPET III (PET+CT) | AutoPET III (FDG+PSMA) |
|---|---|---|---|
| content | head & neck FDG PET/CT | whole-body PET/CT | two tracers |
| size | 883 cases (524 train, 7-9 centers) | 1014 FDG / 900 pts + 597 PSMA / 378 pts | as left |
| labels | GTVp (1) + GTVn (2) | tracer-avid lesions | as left |
| 2nd TEST modality, same patients? | YES — PET registered to CT | YES | **NO — different cohorts/diseases** |
| GT visible in both (needed even if only one is TRAINED)? | YES — contoured on FUSED PET/CT | doubtful — "tracer-avid", i.e. PET-defined | n/a |
| license | challenge registration — verify | CC BY-NC 4.0 (PSMA via TCIA) | as left |

Why HECKTOR wins: its GTV is delineated on the FUSED PET/CT, so the label is by
construction supported in both channels. That matters even under the weaker
"train one, test two" requirement — a PET-trained model TESTED on CT still needs the
tumour to be findable in CT, or the cross-modality number is vacuous rather than hard.
AutoPET's lesions are identified by tracer avidity, so its CT arm — whether trained or
merely tested — is asked to find something partly defined by information CT does not
carry. And AutoPET's FDG/PSMA split is two different patient populations, not two
contrasts of the same patient.

Minimum viable plan with HECKTOR: train on PET only (one arm, like ToothFairy2), test
on PET (in-domain) + CT (cross-modality). If the CT arm proves well-posed, adding a
second CT TRAINING arm is a cheap upgrade that restores full symmetry with the other
tasks.

⚠️ Verify for HECKTOR before committing: **is the CT diagnostic or low-dose
attenuation-correction CT?** PET/CT usually ships low-dose non-contrast CT, on which
head-and-neck tumours are far less conspicuous than on contrast-enhanced diagnostic CT.
That does not make the CT arm vacuous (HECKTOR entrants do gain from the CT channel),
but it may make the CT arm systematically weaker, and that must be understood BEFORE
interpreting a cross-modality gap as a method effect. Check a handful of cases visually
and quantify tumour-to-background contrast on CT vs PET.

Also verify access: HECKTOR is behind challenge registration, which may or may not
still be open — confirm the data is actually obtainable before building anything. If it
is not, fall back to AutoPET PET+CT and treat the weak-CT-arm caveat as a first-class
limitation rather than a footnote.

⚠️ Verify for whichever is chosen, do not assume:
- the exact label semantics (read the definition — AutoPET marks *all* tracer-avid
  lesions with physiological-uptake exclusions; HECKTOR separates GTVp from GTVn and a
  choice must be made whether to train both classes or merge);
- how many studies are lesion-NEGATIVE (AutoPET deliberately includes negative
  controls; an all-empty-GT case behaves differently in Dice and must be handled
  explicitly, cf. toothfairy2's edentulous lower_teeth cases);
- whether PET is stored as SUV or raw counts (normalization depends on it —
  `CTNormalization` is WRONG for PET; z-score or a SUV-aware scheme is likely right,
  and the CT arm and PET arm will need DIFFERENT normalization schemes even though
  they share a grid and a label).

## Recipe — copy ToothFairy2, it is the freshest and cleanest scaffold

`datasets/toothfairy2/5_scripts_toothfairy2/` is the template. Files worth copying
almost verbatim, adapting names:
- `00_utils/env.sh` + `<name>_labels.py` (single source of truth for the label map)
- `00_utils/00_01_audit_source_labels.py` — **run an equivalent FIRST.** On ToothFairy2
  this is what caught that 2 of 5 candidate classes were unannotated in 70% of the
  release. Audit per-source-id, per-cohort, and check EXTENT not just presence.
- `01_create_splits/01_01_create_splits.py` — stratify on any cohort/scanner split
- `02_nnunet/02_00_convert.py`, `02_01_plan_and_preprocess.sh`
- `02_nnunet/ToothFairy2Trainers.py` + `toothfairy2/trainers/*` (registration shim —
  **install it into the venv on EVERY cluster and verify discovery before launching**)
- `04_train/04_01..04_11` (10 wrappers: 6-method suite + 4 ladder rungs)
- `04_train/04_13/04_14_tamia_pack_*.sh`
- `05_predict/05_31_tamia_predict_evaluate.sh` — whole-node predict+eval
- `06_evaluate/06_04_write_configs.py`, `06_05_ladder_summary.py` (thin wrapper over
  the shared `ladder_ood_common.run_ladder_cross_dataset`)

## Gotchas that cost real time this session — do not repeat

1. **Sourced env files must not use short variable names.** `tamia_env_hanseg.sh` set
   bare `HS`/`TF2`, which clobbered the caller's script-dir variables and made an
   ENTIRE evaluation axis record zero tasks while logging the errors to a side file.
   Use `_<NAME>_SCRATCH`. And make the driver HARD FAIL on a short recording.
2. **A QC metric must be shown to respond to the error it is meant to catch.** The
   HaN-Seg MR arm's `tissue_frac` gate turned out ANTI-correlated with accuracy (a
   12 mm-displaced mask scored *better*). Test any new QC by synthetic perturbation —
   see `datasets/hanseg/5_scripts_hanseg/01_prepare/01_03_validate_mr_registration.py`.
3. **A zero-variance statistic is a bug signal, not convergence.** `centroid_shift =
   0.00` on all 42 cases revealed a refinement pass that never ran (a silently-swallowed
   exception).
4. **Verify a patch applied before drawing conclusions from the re-run.** A failed `cd`
   short-circuited a patch via `&&`; the "fixed" re-run was actually the same code, and
   the apparent improvement was run-to-run non-determinism.
5. **ITK registration with RANDOM metric sampling is not reproducible** (multi-threaded;
   the seed does not pin sample-to-thread). Use REGULAR sampling if any registration is
   involved.
6. **Each pack's `RUN_IDS.env` lists ALL six method ids** regardless of which three that
   pack was filtered to train. Compose configs from both suite packs
   (`--suite-pack-a` / `--suite-pack-b`) or a ladder rung silently renders as "—".
7. **Measure srcsm's per-epoch cost on the new dataset.** It was ~1.1x on ToothFairy2,
   not the ~3x of brats/on-harmony. Also: the DualVal/ValSynth trainers were the real
   stragglers (~2x the others) because they run a second validation pass per epoch.
8. **TamIA relay**: `Permission denied (keyboard-interactive)` usually means
   `/dev/shm/<user>-ssh` was wiped by a reboot, NOT a 2FA failure. `mkdir -p -m 700
   /dev/shm/$USER-ssh`, then the user re-runs `ssh -fN tamia.alliancecan.ca`.

## State left behind by this session

- ToothFairy2: 8/10 methods trained (3/3 folds); OURS 2/3 and ladder rung5 0/3, ~2 h
  remaining at handoff. Predict+eval for the 8 done on BOTH axes (24+24 metric files).
- Job `446723` is queued with a dependency on both training chains; it predicts and
  evaluates the stragglers on the CT arm automatically.
- **Still to run after `446723`**: rsync metrics to Vulcan, then
  `06_04_write_configs.sh <suiteA> <suiteB> <ladder>` -> `06_06_cross_dataset_summary.sh`
  -> `06_07_combined_modality_summary.sh` -> `06_05_ladder_summary.sh <A> <B> <L>`.
- HaN-Seg MR arm: built, working, **DISABLED by decision**; its test dirs are renamed
  `_DISABLED_*` on TamIA scratch. Do not re-enable without independent validation.
