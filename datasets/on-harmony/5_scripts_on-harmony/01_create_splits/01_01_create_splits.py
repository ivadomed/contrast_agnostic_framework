#!/usr/bin/env python
"""
Deterministic subject split + scanner exclusion for ON-Harmony benchmark.

Outputs
-------
data/splits/onharmony_splits.json
    4-fold CV in nnUNet format: list of {train: [case_ids], val: [case_ids]}.
    case_id = "sub-{id}_ses-{ses}_T1w"

data/splits/test_cases.json
    List of dicts with subject/session/scanner/t1w/mask/case_id for all test cases.

data/splits/synthseg_labels/fold_{k}/
    Symlinks to SynthSeg masks of training subjects for fold k.
    Consumed by BrainGenerator (SynthSeg-A/B methods) to ensure subject isolation.

Design
------
- 20 subjects, seeded shuffle (seed=12345)
- First 4 after shuffle → test subjects
- Test scanners NOT1ACH and OXF1PRI are excluded from train/val regardless of subject
- Remaining 16 subjects split into 4 folds of 4 subjects each
- Fold k: subjects [4k .. 4k+3] (of the shuffled remaining 16) are val
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = PROJECT_ROOT / "data" / "ON-Harmony"
MASKS_ROOT   = BIDS_ROOT / "derivatives" / "synthseg_masks"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"

TEST_SCANNERS    = {"NOT1ACH", "OXF1PRI"}
N_TEST_SUBJECTS  = 4
N_FOLDS          = 4
SEED             = 12345


def scanner_from_session(ses_name: str) -> str:
    """'ses-NOT2ING001' → 'NOT2ING'"""
    m = re.match(r"^ses-([A-Z][A-Z0-9]+)\d{3}$", ses_name)
    if not m:
        raise ValueError(f"Cannot parse scanner from session name: {ses_name!r}")
    return m.group(1)


def discover_t1w_sessions() -> list[dict]:
    """Walk BIDS tree and collect all T1w sessions that have a SynthSeg mask."""
    sessions = []
    for sub_dir in sorted(BIDS_ROOT.iterdir()):
        if not sub_dir.name.startswith("sub-"):
            continue
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.name.startswith("ses-"):
                continue
            ses = ses_dir.name
            t1w = ses_dir / "anat" / f"{sub}_{ses}_T1w.nii.gz"
            mask = MASKS_ROOT / sub / ses / "anat" / f"{sub}_{ses}_T1w_synthseg.nii.gz"
            if not t1w.exists():
                continue
            if not mask.exists():
                print(f"  WARNING: no SynthSeg mask for {sub} {ses} — skipping")
                continue
            sessions.append({
                "subject":  sub,
                "session":  ses,
                "scanner":  scanner_from_session(ses),
                "t1w":      str(t1w.resolve()),
                "mask":     str(mask.resolve()),
                "case_id":  f"{sub}_{ses}_T1w",
            })
    return sessions


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_sessions = discover_t1w_sessions()
    print(f"Found {len(all_sessions)} T1w sessions with SynthSeg masks")

    all_subjects = sorted({s["subject"] for s in all_sessions})
    assert len(all_subjects) == 20, f"Expected 20 subjects, got {len(all_subjects)}"

    # ── Test subject selection (seeded, deterministic) ──────────────────────────
    rng = random.Random(SEED)
    shuffled = all_subjects.copy()
    rng.shuffle(shuffled)

    test_subjects   = set(shuffled[:N_TEST_SUBJECTS])
    trainval_order  = shuffled[N_TEST_SUBJECTS:]   # 16 subjects in shuffled order

    print(f"\nTest subjects ({N_TEST_SUBJECTS}): {sorted(test_subjects)}")
    print(f"Train/val subjects ({len(trainval_order)}): {sorted(trainval_order)}")

    # ── Categorise sessions ──────────────────────────────────────────────────────
    test_cases = [
        s for s in all_sessions
        if s["subject"] in test_subjects and s["scanner"] in TEST_SCANNERS
    ]
    trainval_cases = [
        s for s in all_sessions
        if s["subject"] not in test_subjects and s["scanner"] not in TEST_SCANNERS
    ]
    discarded = [
        s for s in all_sessions
        if s["subject"] not in test_subjects and s["scanner"] in TEST_SCANNERS
    ]

    print(f"\nTest cases:        {len(test_cases)}")
    print(f"Train/val cases:   {len(trainval_cases)}")
    print(f"Discarded (train/val subject × test scanner): {len(discarded)}")

    # ── 4-fold cross-validation (subject-level isolation) ───────────────────────
    # Split 16 subjects into 4 groups of 4 (preserving shuffled order from above)
    chunk = len(trainval_order) // N_FOLDS
    fold_subjects = [trainval_order[i * chunk:(i + 1) * chunk] for i in range(N_FOLDS)]

    splits = []
    for val_fold in range(N_FOLDS):
        val_subs   = set(fold_subjects[val_fold])
        train_subs = set(trainval_order) - val_subs

        train_ids = sorted(s["case_id"] for s in trainval_cases if s["subject"] in train_subs)
        val_ids   = sorted(s["case_id"] for s in trainval_cases if s["subject"] in val_subs)

        splits.append({"train": train_ids, "val": val_ids})
        print(f"  Fold {val_fold}: {len(train_ids)} train, {len(val_ids)} val "
              f"(val subs: {sorted(val_subs)})")

    # ── Write splits files ───────────────────────────────────────────────────────
    splits_path = SPLITS_DIR / "onharmony_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWritten: {splits_path}")

    test_path = SPLITS_DIR / "test_cases.json"
    with open(test_path, "w") as f:
        json.dump(test_cases, f, indent=2)
    print(f"Written: {test_path}")

    # ── SynthSeg label symlinks per fold ────────────────────────────────────────
    # BrainGenerator (SynthSeg-A/B) must only see training-fold labels.
    for val_fold in range(N_FOLDS):
        val_subs   = set(fold_subjects[val_fold])
        train_subs = set(trainval_order) - val_subs

        link_dir = SPLITS_DIR / "synthseg_labels" / f"fold_{val_fold}"
        link_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for s in trainval_cases:
            if s["subject"] not in train_subs:
                continue
            link = link_dir / f"{s['case_id']}_synthseg.nii.gz"
            target = Path(s["mask"])
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
            n += 1
        print(f"  Fold {val_fold}: {n} SynthSeg label symlinks → {link_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
