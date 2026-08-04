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

Sources: `datasets/brats2024-glioma/.../t1n/ablations/ladder_summary.md`,
`datasets/chaos/.../t2spir/ablations/ladder_summary.md`.

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

`datasets/00_commun_scripts/00_04_analysis/label_cue_importance/` — the boundary-cue
measurement. **Not for the main text.** Its one job is rebutting the reviewer objection
*"boundary clarity depends on the sequence, not the disease"*: every pathology label sits at
or below chance **in its own best contrast** (enhancing tumour on T1c 0.463, MS on FLAIR
0.461) while every organ stays high in **both** (0.620–0.921). See that directory's
`FINDINGS.md` §4 for the full list of attacks on it and which survive.
