# Paper TODO — CURRENT STATUS (2026-10-08, rung-4 lblvor update IN PROGRESS). Everything below the next block is history.

## Rung 4.5 FLAT fill launched (2026-10-08, Paul: noise removal or texture addition?)
- Rung 4.5 = PALETTE with every region at its target mean + PALETTE's label step (no noise, no texture): 4.5->5 isolates
  texture addition exactly; 4->4.5 = noise removal (+ the label-step difference: rung 4 refills labels with Voronoi noise
  cells). Verified before launch (scripts/cluster/rung45_flat/verify_flatfill.py + slice PNGs).
- Training: 13 settings x 3 folds on Vulcan (controllers queued), on-harmony x 3 on TamIA (one H100 node, ~34 h, then rsync
  to Vulcan + scripts/cluster/rung45_flat/post_run.sh). Pancreas not included (not in the paper).
- Analysis ready: paper/scripts/compute_flatfill_decomposition.py --metric dice|hd95 (paper's task-level test on the pairs
  (4,4.5) and (4.5,5); self-check reproduces the current fill-swap numbers). Paper text/figure: TODO once numbers land
  (likely a short paragraph in 4.3 + a 3-segment bar or a row pair in tab:dissociation; decide with Paul).

## Open items from Paul's section-4 review (2026-10-08)
- LATENCY (TODO in 4_experiments.tex, "Training and evaluation"): report PALETTE's training cost (per-iteration GPU time of
  the transform, wall time per fold vs Auglab). The DualVal trainer (synthetic-validation pass) roughly doubles wall time
  (PanSegData H100 probe: OURS DualVal 16.8-17.1 h vs auglab_default 6.2-7.6 h per 2000-epoch fold); state the cost without
  DualVal. Pro-RandConv (CVPR 2023) reports compute + inference time in its discussion: a model for where it goes.
- Spine subject-level split: stated for all tasks in 4.1; verified for the other seven (ON-Harmony: 4 test subjects disjoint
  from the 16 training subjects of all three training modalities), Spine came from a separate pipeline -> confirm.
- HEADS-UP (Paul, 2026-10-08): the ablation narrative will likely change to "training contrast harder than eval contrast
  ==> texture doesn't help much or hurts" once the rung-4 lblvor retrain finishes; Paul has not reviewed the 4.3 results
  prose yet -- don't polish the current one-way-claim wording further until then.
- Naming: the transform is called "PALETTE-Aug" in the abstract/intro/3.1 title, but "PALETTE alone" = transform without Auglab
  and "PALETTE-Aug" = transform inside Auglab (3.2 now defines both). Option: call the transform PALETTE everywhere.

## Rung 4 retrained with label_voronoi (lblvor) -- paper partly updated (2026-10-08)
The noise-fill rung (rung 4) was retrained so noise-refilled labels keep Voronoi cells (`label_voronoi`; old rung 4 gave
each label one noise level, so the fill swap also added Voronoi inside labels). All 18 settings retrain on Vulcan.
- DONE (lblvor rung 4, ladders regenerated): Abdomen T1in/T2spir, Breast T1-CE/T2w (+ the 10 Duke/I-SPY1/ACRIN companion
  ladders), Mandible CBCT, Pelvis CT/MRI. Pancreas T2w done, T1-CE finishing (add_pancreas_test session owns those ladders).
- MS DONE 2026-10-08 16:22 (ladders 06_18/06_19 re-pointed, regenerated: FLAIR swap +10.89, T1w +7.58, task +9.24; MS FLAIR Voronoi
  all-contrast step now flat: 34.6->34.1, p=0.25, was +1.49 p=0.0078).
- PENDING retrain: Glioma (x4), Brain (x3). Regenerate with `LADDER_PENDING=` emptied once they land:
  `scripts/cluster/rung5_val000/run_ladders.sh <keys>` (re-point each script's rung-4 key to the `*_lblvor_*` run first),
  then paper/scripts `compute_dissociation_pvalues.py --metric dice|hd95`, `compute_ladder_significance.py`,
  `make_per_contrast_curves.py`, `make_fill_swap_pairs_table.py` (gen job: $SCRATCH/rung4_lblvor/paper_gen/gen.sh).
- Updated in the .tex: fig:fill-swap-example (lblvor panel) + noise-fill definition (3.3) + experiments clause; fig:ladder /
  fig:ladder-hd95; tab:dissociation(-hd95) rows (pending tasks print \pending; their OLD raw p is still in the 7-task Holm
  family, so every Holm p is provisional); tab:ladder-full (+Voronoi / real-fill cells of pending columns = \pending);
  tab:fill-swap-pairs; tab:mechanism Abdomen +Voronoi 87.2 -> 86.9 (CHAOS-only all-contrast, other columns \pending); the
  Voronoi-is-flat-on-Abdomen-T1in sentence (now significantly negative, p=2e-4); full-ladder prose (Breast T1-CE no longer
  a fill-swap decrease; Mandible's maximum is now the noise-fill rung 69.8); finished-task numbers in abstract/intro/
  experiments/discussion (Breast +0.5 -> +1.5, T1-CE -1.2 -> +0.4, T2w +2.1 -> +2.7; Pelvis CT +4.8 -> +5.9, start 35 -> 34%;
  Abdomen +2.6 -> +3.5; Pelvis +3.5 -> +3.1; Mandible +0.1 -> -1.6 and NOT significant).
- HELD until all ladders land (cross-task claims; the Glioma/MS/Brain deltas will move):
  * abstract/intro/conclusion "at most +3.5 on interface-bounded tasks" (Abdomen is now exactly +3.5) and the MS/Glioma
    task numbers (+6.8/+4.6); intro "four largest gains" list vs "at most +5.9 in any interface-bounded setting" -- Pelvis CT
    +5.86 now sits just below MS T1w +5.94 / Glioma T1n +5.98 (old values): re-check the ranking claim.
  * experiments "each of these tasks passes the correction except Mandible" + "ranking at the top is unchanged" (relative).
  * discussion limitations: "all interface-bounded rows but one start above 63%, every appearance-defined row below 42%";
    permutation test on the seven per-task gains (p=0.2) -- recompute.
  * "Breast gains only slightly" framing (+1.5 now); Brain "only task where the swap significantly hurts".
  * suppl: Voronoi all-contrast check for MS FLAIR / Brain T1w / Glioma T1n; tab:mechanism pending cells (CHAOS-only style
    all-contrast series for MS/Glioma/Brain = all_dice of their within-dataset ladders); fig:ladder-hd95 caption sentence.
  * NGF: noise-fill row recomputed with the lblvor config for all 7 tasks (method key `noisefill_lblvor`,
    config `..._NoiseFillV2LblVorGPUTransform.json` in AugLab); ngf_dose_response_openms_flair still on the OLD rung (its own
    rank CSVs + MS ladder) -- redo after MS lands.

# (previous status, 2026-10-07), evening)

## State
- Builds clean (pdflatex -> bibtex -> pdflatex x2), 0 errors / 0 undefined refs; main text ends on p8 with ~1-2 lines of
  slack (check `endofmaintext` in main.aux after every edit; any addition needs a matching cut).
- Every number regenerated on current data: all 16 ladders on the val000 rung 5 (Brain T1w/T2w landed 10-07), ON-Harmony at
  checkpoint_best/no TTA, SRCSM = the PUBLISHED method (SRC + its test-time source matching; combined + headline configs point
  at `<run>_srcmatch`; plain SRC kept on disk only as an ablation). No \pending / \TODO left in the .tex except Spine.
- Headline (tab:meta): best Dice on 7/8 tasks (all but Mandible), every competitor significantly beaten within task on 5
  (Glioma, MS, Breast, Spine, Pelvis), Abdomen all but SRCSM (85.6 vs 85.9, tie), Mandible: only Auglab significantly ahead,
  PALETTE-Aug third. Cross-task p <= 1.1e-13 vs every competitor. PALETTE alone (7 tasks) beats every competitor and ties
  PALETTE-Aug (p=0.60).
- Fill swap: one Holm family = the seven task-level tests (fable_review's d481d01); Glioma +4.6, MS +6.8, Pelvis +3.5,
  Abdomen +2.6, Breast +0.5, Mandible +0.1, Brain -0.7 Dice (all significant at task level).
- Abstract is Paul's structured format (Introduction/Methods/Results/Conclusion), task-level fill-swap numbers, no cohort or
  test-size talk. Absolute Delta Dice is primary everywhere, relative in brackets (fig:ladder annotation).
- Fig 1 = method in step order (k-means -> Voronoi -> signed affine remap -> label remap); the ablation's noise fill is its own
  figure (fig:fill-swap-example, brain slice, noise vs real with magnified insets).
- Generators: paper/scripts/{compute_dissociation_pvalues, compute_ladder_significance, make_per_contrast_curves,
  make_fill_swap_pairs_table, make_suppl_tables, compute_crosstask_extras, compare_palette_alone_per_task,
  compare_srcsm_srcmatch, generate_method_figure_panels + make_method_figure + make_fill_swap_example_figure}.py.
  Paper task table: bash scripts/evaluate/run_meta_task_heatmap.sh paper/scripts/meta_task_heatmap_paper.yaml.

## Still open
1. Spine: cite the CT/Dixon training cohort and give its training-subject count (two TODO(Paul) comments).
2. Code link for camera-ready (currently "link withheld for review").
3. Optional: measured boundary-definedness predictor (Paul is trying it; would replace the anatomy-assigned grouping).
4. Other sessions edit the paper too (fable_review, 2026-10-07): re-check the numbers above after their commits.

## Resolved 2026-10-07 (detail in git log)
Fable review acted on (`.review/2026-10-07-cvpr-manuscript.md`; archive pooling skipped per Paul); old item 9 (ON-Harmony
val000 == val100) = checkpoint_final sharing, gone at checkpoint_best; bib "template notes" = false positive (never cited);
rung-1 wording (baseline = nnU-Net default aug); GT-crop caveat removed per Paul (crops described by purpose only);
legacy eval configs archived (benchmark/03_archive/legacy_eval_configs_20261007).

# Paper TODO — CURRENT STATUS (2026-10-03). History from 2026-08-02/03 kept below, now mostly superseded.

## Resolved since August (verified 2026-10-03)
- **Page limit**: main text fits 8 pages as of commit 42ab6ee; the 2026-10-03 relative-gain rewrite pushed it ~12 lines over — trimmed in the writing-review pass (check `endofmaintext` in main.aux after every build).
- **Ladder-format unification (§3b)**: done — every ladder goes through `ladder_ood_common.py`; `tab:dissociation`/`tab:ladder-full` are generated by `paper/scripts/compute_dissociation_pvalues.py` / `compute_ladder_significance.py` from `ladder_series.json`.
- **val100 supplementary (§4.3)**: done — `tab:suppl-val100` (cross-task two-sided Dice p=0.74, HD95 p=0.096).
- **Dataset roster (§4.4)**: done — 8 tasks in `tab:meta` (spine + pelvis added); eval-only cohorts listed in `_suppl_data.tex`.
- **Spine task (§5)**: done (healthy-spine-tum, no SRCSM arm).
- **10-panel ladder figure / visual check**: superseded by the 2x3 per-contrast figure (`make_per_contrast_curves.py`), visually checked 2026-10-03.
- **Dissociation framing**: 14 rows (all training modalities); claim is now a GRADED one-way result stated RELATIVE to the noise-fill Dice (Paul, 2026-10-03): BraTS-GLI +14.8%, Open-MS +15.8% vs CHAOS +3.9%, ToothFairy2 +2.0%, ON-Harmony −3.0%; Breast −0.5% is the stated exception.
- **CHAOS**: external CT/MR cropped to the CHAOS FOV before prediction (main table 5/5); CHAOS ladders cross-dataset (AMOS + SLIVER07) with retrained rung 5.

## Still open
1. **Rung-5 retrains for ON-Harmony T1w and BraTS-GLI T2w** (TamIA chain 502645-9, predict 502888, eval 502889): when metrics land, point `06_10` (on-harmony T1w) / `06_14` (brats t2w) at the new run ids, re-run the ladders, then `compute_dissociation_pvalues.py`, `compute_ladder_significance.py`, `make_per_contrast_curves.py`, and update tab:dissociation, tab:ladder-full, fig:ladder caption and every per-task % in abstract/intro/results/conclusion. CHAOS T2spir's same fix moved it from −0.88 to +5.11, so expect these rows to move.
2. **Rung 6 (PALETTE + boundary PV)**: training on TamIA; not in the paper yet (Paul: in 1-2 days).
3. **Bibliography**: several entries print private annotation notes in the reference list; incomplete author lists from the August additions — see the bibliography audit.
4. **Reviewer objection to pre-empt (Paul's call)**: relative gain (Δ / noise-fill Dice) favours low-baseline tasks; the remaining-error view (Δ / (100 − noise-fill Dice)) would rank CHAOS (+20%) above BraTS-GLI (+8%). The limitations paragraph states the absolute CHAOS T2spir number; decide whether to say more.
5. `sec/_suppl_mechanism.tex` has a 103pt overfull box (supplement, pre-existing).
6. **Spine training cohort is uncited** (`% TODO(Paul)` in 4_experiments.tex Datasets).
7. **Test-time crops use the reference masks** (CHAOS slab anchored on GT kidneys/liver; breast lesion-side crop extended to contain the lesion) — identical for all methods, but not disclosed as GT-informed. Decide whether to state it.
8. **Baseline fairness details a reviewer will ask for:** which label map SynthSeg-noEM got on tasks without a dense parcellation; whether SRCSM's test-time histogram matching was on (SRCSM 16.4 on Breast will read as misconfiguration); SynthSeg-EM "described in the original work" attribution not verified against the 2023 MedIA paper.
9. **ON-Harmony T1w val000 vs val100 ladder rungs are near-identical** (per-case Dice within ~1e-4, though the two checkpoint_best files differ) — suspect both mirrors were predicted from the same checkpoint (checkpoint_final?). Affects tab:ladder-full v100 cells (T1w, DWI) and possibly tab:suppl-val100's ON-Harmony row. Unverified.
10. Task naming: tables use Mandible/Breast, ladder tables/figure use ToothFairy2/I-SPY2 — left as is.
11. `sec/_suppl_ngf.tex` is dead (not \input) but duplicates labels with stale numbers — move to toDelete/ when convenient.

---

# Paper work session 2026-08-02 — what changed, what's verified, what's missing

## STATUS AT END OF SESSION: compiles clean, fits the limit, supplementary written

`paper/PALETTE_draft_20260802.pdf` — **13 pages: main text ends p8** (CVPR 2026 limit is 8,
references excluded), references pp. 9–10, **supplementary pp. 11–13**.
Verified by a from-scratch rebuild (pdflatex → bibtex → pdflatex ×2): exit 0, **0 undefined
references, 0 undefined citations, 0 LaTeX errors, 0 BibTeX errors**, 1 overfull box.

| | main text | supplementary |
|---|---|---|
| figures | method pipeline; ablation ladder (`fig:ladder`) | NGF dose-response (`fig:ngf-dose`) |
| tables | task-level results (`tab:meta`); the 4-task dissociation (`tab:dissociation`) | per-setting Dice + HD95 (8 settings); val000-vs-val100; component ablation; NGF |

Moved out of the main text to fit: the component ablation and the NGF measurement, both now
supplementary sections referenced from a single "Supporting analyses" paragraph.


All edits keep a `*.bak_20260802` sibling. Nothing was deleted.

---

## 1. Verified facts (checked this session, not assumed)

| fact | status |
|---|---|
| **CVPR 2026 page limit = 8 pages** excluding references | ✅ confirmed on the [official author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines). You assumed 7 — you have a page more than you thought. |
| **"No competitor is reliably second"** | ✅ **verified from the metrics.** Best rival by training set: Auglab on brats t1n/t2w, chaos t1in/t2spir (Dice), on-harmony T1w, open-ms t1w; **SynthSeg-EM** on on-harmony T2w and chaos t2spir (HD95); **SRCSM** on open-ms flair (Dice) and chaos t1in / open-ms t1w (HD95). Three distinct runners-up on each metric. |
| **PALETTE significantly better than all 5 competitors, both metrics** | ✅ from `meta_task_heatmap_summary.md`: Dice p = 1.5e-38 … 3e-4; HD95 p = 2.6e-30 … 0.022. The old "significant in six of eight" **undersold the result** and has been replaced. |
| **The texture dissociation is now 2-vs-2** | ✅ **both new ladders finished.** See §2. |

## 2. The big result that landed while you were away

Both outstanding ablation ladders completed. The rung 4→5 fill swap (noise→real,
partition held fixed) now has two tasks on each side:

| task | target type | Δ Dice | Δ HD95 (corrected, see §3b) |
|---|---|---:|---:|
| Open-MS FLAIR | appearance-defined | **+7.70** | −3.05 |
| **Brats-GLI T1n** (new) | appearance-defined | **+7.22** | −4.59 |
| CHAOS T1in | interface-defined | +1.07 | −2.97 |
| **CHAOS T2spir** (new) | interface-defined | **−1.14** | +4.20 |

Note the dissociation holds on **Dice only** — see §3b. Do not describe HD95 as corroborating it.

Sources: `benchmark/02_tasks/brain_tumor/brats2024-glioma/.../t1n/ablations/ladder_summary.md`,
`benchmark/02_tasks/abdomen_healthy/chaos/.../t2spir/ablations/ladder_summary.md`.

This is the control arm the paper needed. The effect is +7.2/+7.7 on appearance-defined
targets and within noise **with inconsistent sign** on interface-defined ones. It is now
the paper's central result (§`subsec:e-ablation`, `tab:dissociation`).

## 3. What I changed

- **`0_abstract.tex`** — rewritten mechanism-first. Opens with the question ("we ask when
  that matters, and the answer is a property of the task"), states the appearance-vs-interface
  distinction, leads with the dissociation, and claims *"matches or exceeds the best competing
  method on every task, and improves significantly over all five competitors"*.
- **`1_intro.tex`** — contributions reordered: (1) the task-level account, (2) the causal
  ablation, (3) PALETTE, (4) breadth evidence.
- **`4_experiments.tex`**
  - `Main comparison` rebuilt around the **task-level meta table** (`tab:meta`, Dice + HD95,
    6 methods × 4 tasks + overall + p). Old `tab:main` had **stale numbers** (it claimed
    Open-MS Ours 41.7; current value is 38.5 — the test sets grew). Per-setting tables now
    point to supplementary.
  - New paragraphs: *"No competitor is reliably second"* and *"Reading the non-significant
    cells"* (equivalence, not shortfall — the sets are large).
  - `Causal ablation` rewritten around `tab:dissociation` (the 2-vs-2 above).
- **`5_discussion.tex`** — deleted the *"Aggregate margins are thin by design"* paragraph
  (an unforced concession, and now inaccurate given task-level significance); replaced with
  the consistency claim. Added *"Why the dissociation follows from the anatomy"*.
- **`main.bib`** — added `lee2007gaclivers`, `gliomamargins2025`, `flairectomy2022`,
  `zhang2008mstexture`, `visser2019interrater`, `commowick2018msseg`.
- **`NARRATIVE.md`** — addendum with the reframe, the drop-in "why" paragraph, and three
  traps not to simplify back in.

## 3b. ⚠️ ERROR FOUND AND FIXED in the dissociation table (2026-08-02, late)

Verifying `tab:dissociation` against source revealed the **HD95 column was wrong for the two
older ladders**. The Dice column verified exactly; HD95 did not:

| task | was | recomputed from source (OOD-contrast mean) |
|---|---|---|
| Open-MS FLAIR | −4.77 | **−3.05** |
| CHAOS T1in | −0.72 | **−2.97** |

The −0.72 was a mis-transcription: it is the *Dice* delta of the `+voronoi` rung in the CHAOS
**T2spir** ladder. Both are corrected in the paper.

**This changed a claim.** With the correct numbers, HD95 does **not** show the dissociation:
boundary distance improves on three of four tasks, including interface-defined CHAOS T1in
(−2.97 mm), and degrades only on CHAOS T2spir. The paper now restricts the dissociation claim
to **Dice** and states the HD95 behaviour explicitly rather than implying it corroborates.

**Root cause worth fixing properly:** the four ladders do not share a summary format. BraTS T1n
and CHAOS T2spir have proper `ablations/ladder_summary.md` files with explicit rung tables;
Open-MS FLAIR and CHAOS T1in only have per-contrast `*_summary.md` files, and their OOD rung
values must be recomputed by hand (average the non-training contrast columns — verified: this
reproduces 19.57→27.27 for Open-MS and 87.15→88.22 for CHAOS T1in exactly).
**Regenerate all four with the same script before submission** so the central table is provably
computed one way.

## 4. ⚠️ MUST FIX before submission

1. ~~Full numeric audit~~ **DONE.** Fixed: "four methods"→five (SRCSM added as baseline 4);
   the wrong single "1000 epochs" schedule → the real per-dataset budget (2500/2000/2000/200)
   plus 3-fold; the statistics paragraph now describes the *actual* task-level test (stratified
   sign-flip permutation on macro-averaged per-case differences) rather than only Wilcoxon;
   stale "+8.9"→"+6.7" with the live numbers; "six/seven of eight"→"every task"; the dangling
   "result above"; the unfulfilled promise of per-structure breakdowns; and the conclusion
   rewritten mechanism-first. **Remaining:** the ladder-format unification in §3b.
2. **Incomplete bib entries.** The six I added have notes and enough to identify them but
   several lack full author lists / DOIs. Complete them.
3. **`val100` needs its supplementary section.** Decision made: **val000 is the headline**
   (all competitors are val000, so it is the fair comparison); val100 goes to supplementary
   as "PALETTE can also be used during validation, improving results further" — it is
   statistically indistinguishable from val000 at task level (Dice p=0.66, HD95 p=0.99), so
   frame it as a free extra, not a second method.
4. **Decide the dataset roster.** The meta table covers **4 training tasks**. The repo has
   many more evaluation-only sets (ms3seg, mslesseg, amos, sliver07, msd-spleen,
   cirrmri-liver, kidney-t2w, trusted, brats-ssa2024) and you mentioned a **spine** task
   that does not exist in the results tree yet. Either fold them into the meta table or
   state explicitly in the setup what is in and what is out — a reviewer will ask why some
   datasets appear only in supplementary.
5. **`tab:meta` bolding.** With val100 removed from the main table, PALETTE is best in every
   Dice column but **not** on HD95 for Brats (Auglab 15.0 vs ours 15.2) or CHAOS (SRCSM 23.9
   vs ours 24.3). I have bolded honestly. Do not "fix" this — it is what makes the
   *consistency* argument credible, and both are within noise.

## 5. Experiments worth running

| priority | experiment | why |
|---|---|---|
| **high** | **Spine task** (you mentioned it; absent from the results tree) | A third interface-defined task would make the dissociation 2-vs-3 and is the cheapest way to strengthen the paper's central claim |
| **high** | Re-derive every per-setting number for the supplementary tables | See §4.1 — correctness risk, not a new experiment |
| medium | on-harmony ladder | **Currently running** (TamIA job 392288, ~22 h in, 2 pending links). Would add a 3rd appearance/interface data point — on-harmony's targets are brain anatomy, so predict a *small* effect |
| medium | Per-contrast breakdown of "no competitor is reliably second" | Verified at task level; a per-contrast version would be a stronger supplementary figure |
| low | Anisotropy control for the `label_cue_importance` analysis | Only needed if you put that analysis in the paper. Currently held in reserve |

## 6. Held in reserve (not in the paper)

`benchmark/00_commun_scripts/00_04_analysis/label_cue_importance/` — the boundary-cue
measurement. **Not for the main text.** Its one job is rebutting the reviewer objection
*"boundary clarity depends on the sequence, not the disease"*: every pathology label sits at
or below chance **in its own best contrast** (enhancing tumour on T1c 0.463, MS on FLAIR
0.461) while every organ stays high in **both** (0.620–0.921). See that directory's
`FINDINGS.md` §4 for the full list of attacks on it and which survive.

---

## Session addendum 2026-08-03 (part 2): ON-Harmony ablation + ladder unification + 10-panel figure

### ON-Harmony added as a third, distinct pattern (not forced into the binary)
`tab:dissociation` now has 5 rows across all 4 headline tasks (CHAOS twice):
Dice fill-swap effect is **+7.70 / +7.22** (no tissue interface), **+1.07 / −1.14**
(tissue interface, near-null/inconsistent sign), and **−4.48** (ON-Harmony,
densely-labelled healthy brain) — large in magnitude but the *opposite* sign
of the positive cases, fitting neither existing category. Two candidate,
explicitly-hedged (not asserted) explanations are given in
`subsec:e-ablation`: (1) ON-Harmony's 31 labels parcellate ~the whole brain
volume (verified against `dataset.json`), unlike the few localized labels of
the other tasks, so the per-label remap step already performs a near-dense,
SynthSeg-like resampling even under noise fill — real-fill on top of that may
encode silver-standard/registration noise as if it were signal; (2) brain
tissue boundaries may sit nearer the tissue-interface end of the spectrum than
tumour/lesion margins, though the effect size argues against a purely-null
organ-like account. Every location that previously implied a clean 2-vs-2
split (abstract, intro ×2, discussion, conclusion) has been corrected to
reflect the 3-pattern reality; none now overclaim.

### Ladder scripts unified (root cause from §3b finally fixed)
New shared engine: `benchmark/00_commun_scripts/00_03_evaluate/ladder_ood_common.py`.
- Retrofitted BraTS-T1n (06_13), CHAOS-T2spir (06_33), ON-Harmony-T1w (06_10) as
  thin wrappers — **regression-verified**: identical numbers to their previous
  independent implementations.
- Added CHAOS-T1in (06_34, new) and Open-MS-FLAIR-OOD (06_14, new) — these two
  previously had no matching script (Open-MS's only existing 06_12 is a
  different, older format with no HD95/OOD split). Running the canonical
  engine against them caught **two more small HD95 discrepancies** vs. my
  earlier hand-reconstruction: CHAOS T1in −2.97→**−2.95**, Open-MS FLAIR
  −3.05→**−3.02**. Both corrected in the paper. Dice values matched exactly.
- All five now write `ladder_series.json` (machine-readable) alongside the
  existing `ladder_summary.md` + 2-panel PNG.

### New 10-panel main-text figure
`paper/make_ladder_panels.py` reads all 5 `ladder_series.json` files and emits
10 individual small PDFs (`figures/ladder_panels/*.pdf`) — one Dice + one HD95
trajectory per task, color-coded by boundary type (red = no interface, blue =
tissue interface, purple = dense label map), fill-swap rung highlighted
consistently. Wired into `\cref{fig:ladder}` as a proper `figure*` with 10
`subfigure` environments (2 rows × 5 cols: Dice row, then HD95 row), replacing
the old hand-assembled two-task composite PNG.

**⚠️ Not visually verified.** The Read/image tool was unavailable for the
entire second half of this session (repeated hook timeouts, not something I
could route around) — verification is structural only: clean `pdflatex`
compile, 0 new overfull/underfull warnings (the pre-existing one is in
`_suppl_tables.tex`, unrelated), and all 10 expected panel titles present in
the extracted text layer at the correct positions. **Recommend a visual
spot-check of page 8 (`fig:ladder`) before trusting the layout** — subfigure
widths (0.19\linewidth × 5 with tiny 5.2pt tick labels) were sized by
calculation, not by looking at the result.

### Page budget got worse, not better
Main text is now **10 pages** (was 9 before this addendum), against the
8-page limit and your ~7-page target. The new full-width figure is a
meaningful contributor. This is now the single most pressing open item.

---

## Session addendum 2026-08-03 (part 2): ON-Harmony ablation + ladder unification + 10-panel figure

### ON-Harmony added as a third, distinct pattern (not forced into the binary)
tab:dissociation now has 5 rows across all 4 headline tasks (CHAOS twice):
Dice fill-swap effect is +7.70 / +7.22 (no tissue interface), +1.07 / -1.14
(tissue interface, near-null/inconsistent sign), and -4.48 (ON-Harmony,
densely-labelled healthy brain) -- large in magnitude but the opposite sign
of the positive cases, fitting neither existing category. Two candidate,
explicitly-hedged (not asserted) explanations are given in subsec:e-ablation:
(1) ON-Harmony's 31 labels parcellate ~the whole brain volume (verified
against dataset.json), unlike the few localized labels of the other tasks,
so the per-label remap step already performs a near-dense, SynthSeg-like
resampling even under noise fill -- real-fill on top of that may encode
silver-standard/registration noise as if it were signal; (2) brain tissue
boundaries may sit nearer the tissue-interface end of the spectrum than
tumour/lesion margins, though the effect size argues against a purely-null
organ-like account. Every location that previously implied a clean 2-vs-2
split (abstract, intro x2, discussion, conclusion) has been corrected to
reflect the 3-pattern reality; none now overclaim.

### Ladder scripts unified (root cause from earlier addendum's section 3b finally fixed)
New shared engine: benchmark/00_commun_scripts/00_03_evaluate/ladder_ood_common.py.
- Retrofitted BraTS-T1n (06_13), CHAOS-T2spir (06_33), ON-Harmony-T1w (06_10) as
  thin wrappers -- regression-verified: identical numbers to their previous
  independent implementations.
- Added CHAOS-T1in (06_34, new) and Open-MS-FLAIR-OOD (06_14, new) -- these two
  previously had no matching script (Open-MS's only existing 06_12 is a
  different, older format with no HD95/OOD split). Running the canonical
  engine against them caught two more small HD95 discrepancies vs. my
  earlier hand-reconstruction: CHAOS T1in -2.97 -> -2.95, Open-MS FLAIR
  -3.05 -> -3.02. Both corrected in the paper. Dice values matched exactly.
- All five now write ladder_series.json (machine-readable) alongside the
  existing ladder_summary.md + 2-panel PNG.

### New 10-panel main-text figure
paper/make_ladder_panels.py reads all 5 ladder_series.json files and emits
10 individual small PDFs (figures/ladder_panels/*.pdf) -- one Dice + one HD95
trajectory per task, color-coded by boundary type (red = no interface, blue =
tissue interface, purple = dense label map), fill-swap rung highlighted
consistently. Wired into fig:ladder as a proper figure* with 10 subfigure
environments (2 rows x 5 cols: Dice row, then HD95 row), replacing the old
hand-assembled two-task composite PNG.

NOT VISUALLY VERIFIED. The Read/image tool was unavailable for the entire
second half of this session (repeated hook timeouts, not something routable
around) -- verification is structural only: clean pdflatex compile, 0 new
overfull/underfull warnings (the pre-existing one is in _suppl_tables.tex,
unrelated), and all 10 expected panel titles present in the extracted text
layer at the correct positions. Recommend a visual spot-check of page 8
(fig:ladder) before trusting the layout -- subfigure widths (0.19 linewidth
x5 with tiny 5.2pt tick labels) were sized by calculation, not by looking at
the result.

### Page budget got worse, not better
Main text is now 10 pages (was 9 before this addendum), against the 8-page
limit and the ~7-page target. The new full-width figure is a meaningful
contributor. This is now the single most pressing open item.
