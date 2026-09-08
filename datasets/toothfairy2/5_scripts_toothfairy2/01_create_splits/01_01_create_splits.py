#!/usr/bin/env python3
"""
ToothFairy2 train/test partition + 3-fold CV splits.

One CBCT volume = one patient = one nnU-Net case, so there is no grouping hazard
here (contrast ispy2, where both FOV variants of a patient had to stay in the same
split, or chaos, where two contrasts share a patient partition). That makes this
script simple, and the simplicity is the point — the only real decision is which
cases to seal away as the held-out test set.

STRATIFICATION: COHORT, THEN DENTITION BURDEN
---------------------------------------------
1. COHORT (F vs P). ToothFairy2 is two acquisition cohorts, not one: the 63
   `ToothFairy2F_*` volumes have a median 82 mm z field-of-view, the 417
   `ToothFairy2P_*` volumes only 51 mm (see 00_utils/toothfairy2_labels.py's
   docstring for the full audit). That is a large, real domain shift in how much
   anatomy is imaged — and F is only 13% of the release, so a plain random draw can
   easily leave a fold or the test set with almost no F cases. Patients are split
   WITHIN each cohort and the per-cohort folds merged, so every fold and the test set
   get a proportional share of both. Same two-level scheme ispy2 uses for laterality.

2. DENTITION BURDEN, within each cohort — lower-teeth voxel count, from the BIDS
   conversion audit (no image I/O needed here). This is the dominant source of
   between-patient variation in the reduced task: a fully dentate patient and an
   edentulous one differ enormously in how much label volume `lower_teeth` occupies
   and in how the mandibular alveolar ridge presents (resorption). Test cases are
   taken at evenly-spaced burden ranks so the held-out set spans low->high burden
   instead of landing all-dentate or all-edentulous — which would then read as a
   method effect. Same evenly-spaced-rank approach ambl / atlas-liver-hcc / ispy2
   use for tumour burden.

FOLDS: 3, per CLAUDE.md's permanent fold policy (folds 0 1 2 only). This is a new
split, so it emits exactly 3 — no legacy 4th fold that nobody will ever train.

OUTPUTS (4_splits_toothfairy2/)
  partition.json     {"train_pool_cases", "test_cases", per-case teeth burden}
  splits_final.json  3-fold CV over the train-pool cases
  test_cases.json    {"test": [...]}  — read by the trainer contamination guard
  splits_report.md   fold sizes, burden balance, sanity checks

Run:  bash 01_create_splits/01_01_create_splits.sh
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from splits_lib import kfold_splits, write_splits_final  # noqa: E402

SEED = 1337
TEST_FRACTION = 0.15
N_FOLDS = 3
# Above this share of unusable cases, stop instead of excluding: that many failures
# at once means something systematic, not a handful of degenerate volumes.
MAX_EXCLUDED_FRACTION = 0.05

# Read from the environment, with the git-repo-relative layout as fallback. These
# MUST honour the env: on TamIA every bulk path is redirected to $SCRATCH by
# scripts/cluster/tamia_env_toothfairy2.sh, and hardcoding the repo-relative
# defaults here would silently write the splits (and the preprocessed-dir copy) to
# $PROJECT instead — i.e. training would read a DIFFERENT splits file from the one
# this script just reported on, or none at all.
BIDS_ROOT = Path(os.environ.get(
    "BIDS_ROOT", DATASET_ROOT / "1_BIDS_toothfairy2" / "maxillofacial-toothfairy2"))
SPLITS = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_toothfairy2"))
NNUNET_PRE = Path(os.environ.get(
    "nnUNet_preprocessed", DATASET_ROOT / "2_nnUNet_toothfairy2" / "preprocessed"))
DS_NAME = os.environ.get("NNUNET_DATASET_ID", "Dataset110_ToothFairy2CBCT")


def case_id(sub: str) -> str:
    return f"toothfairy2_{sub.removeprefix('sub-')}"


def stratified_pick(pool: list[str], burden: dict[str, float], k: int) -> list[str]:
    """k cases at evenly-spaced burden ranks (deterministic; tops up on ties)."""
    by = sorted(pool, key=lambda c: (burden[c], c))
    idx = np.linspace(0, len(by) - 1, k).round().astype(int)
    picked = {by[i] for i in idx}
    i = 0
    while len(picked) < k and i < len(by):
        picked.add(by[i]); i += 1
    return sorted(picked)


def main() -> None:
    audit_p = BIDS_ROOT / "conversion_audit.json"
    if not audit_p.exists():
        raise SystemExit(f"{audit_p} missing — run 00_utils/00_00_extract_and_bidsify.sh first")
    audit = json.loads(audit_p.read_text())
    if audit.get("n_failed"):
        raise SystemExit(f"conversion had {audit['n_failed']} failures — fix before splitting")

    rows = audit["cases"]
    burden = {case_id(r["sub"]): float(r["label_voxels"]["lower_teeth"]) for r in rows}
    cohort = {case_id(r["sub"]): ("F" if "F_" in r["case"] else "P") for r in rows}
    cases = sorted(burden)
    print(f"{len(cases)} cases; teeth-voxel burden "
          f"min={min(burden.values()):.0f} max={max(burden.values()):.0f}")

    # ── usability guard: EXCLUDE-AND-RECORD, not hard-fail ────────────────────
    # A case with no mandible at all is not a usable maxillofacial volume (truncated
    # FOV, or an annotation that covers only the airway — ToothFairy2P_077 is exactly
    # this). One degenerate volume must NOT halt an unattended pipeline, so such
    # cases are dropped and recorded in partition.json. But a SYSTEMATIC breakage
    # (a bad conversion, a wrong label map) would show up as many exclusions at
    # once, and that still must stop everything — hence the threshold.
    excluded = sorted(case_id(r["sub"]) for r in rows
                      if r["label_voxels"]["mandible"] == 0)
    if excluded:
        frac = len(excluded) / len(cases)
        print(f"excluding {len(excluded)} case(s) with no mandible label "
              f"({frac:.1%}): {excluded[:10]}")
        if frac > MAX_EXCLUDED_FRACTION:
            raise SystemExit(
                f"{frac:.1%} of cases have no mandible — above the "
                f"{MAX_EXCLUDED_FRACTION:.0%} threshold. This looks systematic "
                f"(bad conversion or wrong label map), not a few degenerate volumes. "
                f"Investigate before splitting.")
        drop = set(excluded)
        cases = [c for c in cases if c not in drop]
        burden = {c: burden[c] for c in cases}

    # ── test hold-out: stratified by cohort, then by burden within cohort ──────
    test: list[str] = []
    for coh in sorted(set(cohort[c] for c in cases)):
        members = [c for c in cases if cohort[c] == coh]
        k = round(len(members) * TEST_FRACTION)
        test += stratified_pick(members, burden, k)
    test = sorted(test)
    pool = [c for c in cases if c not in set(test)]

    # ── CV folds: k-fold WITHIN each cohort, then merged, so each fold carries a
    # proportional share of the small F cohort instead of leaving some fold without
    # any wide-FOV cases at all.
    rng = np.random.default_rng(SEED)
    per_cohort_splits = []
    for coh in sorted(set(cohort[c] for c in pool)):
        members = [c for c in pool if cohort[c] == coh]
        rng.shuffle(members)
        per_cohort_splits.append(kfold_splits(members, N_FOLDS))
    splits = []
    for k in range(N_FOLDS):
        splits.append({
            "train": sorted(c for ps in per_cohort_splits for c in ps[k]["train"]),
            "val":   sorted(c for ps in per_cohort_splits for c in ps[k]["val"]),
        })

    # ── invariants, re-checked from the emitted lists (not from construction) ──
    test_set = set(test)
    for k, s in enumerate(splits):
        assert not (set(s["train"]) & set(s["val"])), f"fold {k}: train/val overlap"
        leak = test_set & (set(s["train"]) | set(s["val"]))
        assert not leak, f"fold {k}: held-out test leaked into CV: {sorted(leak)[:5]}"
    assert sorted(set().union(*(set(s["val"]) for s in splits))) == sorted(pool), \
        "union of val folds != train pool"

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps(
        {"seed": SEED, "test_fraction": TEST_FRACTION, "n_folds": N_FOLDS,
         "cohorts": {c: cohort[c] for c in cases},
         "excluded_cases": excluded,
         "excluded_reason": "no mandible label (unusable maxillofacial volume)",
         "train_pool_cases": sorted(pool), "test_cases": test,
         "teeth_voxels": {c: burden[c] for c in cases}}, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({"test": test}, indent=2))
    write_splits_final(splits, SPLITS, NNUNET_PRE, DS_NAME)

    def med(cs): return float(np.median([burden[c] for c in cs])) if cs else 0.0
    def nF(cs): return sum(1 for c in cs if cohort[c] == "F")
    lines = ["# ToothFairy2 splits", "",
             f"- cases: {len(cases)}  test: {len(test)}  train-pool: {len(pool)}",
             f"- excluded (no mandible): {len(excluded)} {excluded if excluded else ''}",
             f"- folds: {N_FOLDS} (CLAUDE.md fold policy: 0 1 2 only)",
             f"- stratifier: cohort (F/P) then lower-teeth voxel count; "
             f"median burden test={med(test):.0f} "
             f"pool={med(pool):.0f}", "",
             f"- test set: {len(test)} cases ({nF(test)} F / {len(test) - nF(test)} P)",
             "",
             "| fold | train | val | F in val | median lower-teeth voxels (val) |",
             "|---|---|---|---|---|"]
    for k, sp in enumerate(splits):
        lines.append(f"| {k} | {len(sp['train'])} | {len(sp['val'])} | {nF(sp['val'])} "
                     f"| {med(sp['val']):.0f} |")
    (SPLITS / "splits_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote partition.json / splits_final.json / test_cases.json -> {SPLITS}")


if __name__ == "__main__":
    main()
