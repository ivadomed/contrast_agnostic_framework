#!/usr/bin/env python
"""
Deterministic subject split + scanner exclusion for ON-Harmony T2w benchmark.

Mirrors 01_01_create_splits.py's subject-level test/fold assignment exactly (same
seed, same shuffle over the same 20 subjects) so T1w and T2w share identical
train/val/test subject groups — only the per-subject session set differs (T2w has
fewer/more sessions per subject than T1w).

Outputs (under 4_splits_on-harmony/, mirroring onharmony_splits.json/test_cases.json)
-------
onharmony_t2w_splits.json
    4-fold CV in nnUNet format: list of {train: [case_ids], val: [case_ids]}.
    case_id = "sub-{id}_ses-{ses}_T2w"
test_cases_t2w.json
    List of dicts with subject/session/scanner/t2w/mask/case_id for all test cases.

Usage: .venv/bin/python 01_02_create_splits_t2w.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "00_utils"))

DATASET_ROOT = SCRIPT_DIR.parents[1]
BIDS_ROOT  = Path(os.environ.get("BIDS_ROOT", DATASET_ROOT / "1_BIDS_on-harmony"))
MASKS_ROOT = BIDS_ROOT / "derivatives" / "synthseg_masks"
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_on-harmony"))

TEST_SCANNERS    = {"NOT1ACH", "OXF1PRI"}
N_TEST_SUBJECTS  = 4
N_FOLDS          = 4
SEED             = 12345    # must match 01_01_create_splits.py for identical subject groups


def scanner_from_session(ses_name: str) -> str:
    """'ses-NOT2ING001' → 'NOT2ING'"""
    m = re.match(r"^ses-([A-Z][A-Z0-9]+)\d{3}$", ses_name)
    if not m:
        raise ValueError(f"Cannot parse scanner from session name: {ses_name!r}")
    return m.group(1)


def discover_t2w_sessions() -> list[dict]:
    """Walk BIDS tree and collect all T2w sessions that have a SynthSeg mask."""
    sessions = []
    for sub_dir in sorted(BIDS_ROOT.iterdir()):
        if not sub_dir.name.startswith("sub-"):
            continue
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.name.startswith("ses-"):
                continue
            ses = ses_dir.name
            t2w = ses_dir / "anat" / f"{sub}_{ses}_T2w.nii.gz"
            mask = MASKS_ROOT / sub / ses / "anat" / f"{sub}_{ses}_T2w_synthseg.nii.gz"
            if not t2w.exists():
                continue
            if not mask.exists():
                print(f"  WARNING: no SynthSeg mask for {sub} {ses} T2w — skipping")
                continue
            sessions.append({
                "subject":  sub,
                "session":  ses,
                "scanner":  scanner_from_session(ses),
                "t2w":      str(t2w.resolve()),
                "mask":     str(mask.resolve()),
                "case_id":  f"{sub}_{ses}_T2w",
            })
    return sessions


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_sessions = discover_t2w_sessions()
    print(f"Found {len(all_sessions)} T2w sessions with SynthSeg masks")

    all_subjects = sorted({s["subject"] for s in all_sessions})
    assert len(all_subjects) == 20, f"Expected 20 subjects, got {len(all_subjects)}"

    # ── Test subject selection — IDENTICAL shuffle/seed to 01_01, so subject
    # groups match the T1w splits exactly. ──────────────────────────────────────
    rng = random.Random(SEED)
    shuffled = all_subjects.copy()
    rng.shuffle(shuffled)

    test_subjects  = set(shuffled[:N_TEST_SUBJECTS])
    trainval_order = shuffled[N_TEST_SUBJECTS:]

    print(f"\nTest subjects ({N_TEST_SUBJECTS}): {sorted(test_subjects)}")
    print(f"Train/val subjects ({len(trainval_order)}): {sorted(trainval_order)}")

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

    splits_path = SPLITS_DIR / "onharmony_t2w_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWritten: {splits_path}")

    test_path = SPLITS_DIR / "test_cases_t2w.json"
    with open(test_path, "w") as f:
        json.dump(test_cases, f, indent=2)
    print(f"Written: {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
