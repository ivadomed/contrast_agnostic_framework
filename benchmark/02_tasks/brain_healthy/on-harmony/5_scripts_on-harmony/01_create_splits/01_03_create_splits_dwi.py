#!/usr/bin/env python
"""
Deterministic subject split + scanner exclusion for ON-Harmony dwi_ap benchmark
(3rd training modality, added 2026-09-21 to regain independent-patient power on
the srcsm-vs-Ours combined comparison discussion — see project memory
project_onharmony_significance_rerun_20260921 / advisor note same day).

Mirrors 01_01_create_splits.py's subject-level test/fold assignment exactly (same
seed, same shuffle over the same 20 subjects) so T1w/T2w/dwi_ap share identical
train/val/test subject groups — only the per-subject session set differs (not
every session has a usable dwi_ap acquisition + registered mask).

Outputs (under 4_splits_on-harmony/, mirroring onharmony_t2w_splits.json/test_cases_t2w.json)
-------
onharmony_dwi_splits.json
    4-fold CV in nnUNet format: list of {train: [case_ids], val: [case_ids]}.
    case_id = "sub-{id}_ses-{ses}_dwi_ap"
(no separate test_cases_dwi.json: 06_01_evaluate_testset.sh's shared cross-contrast
test-set builder already reads test_cases.json (T1w's) as the master (subject,
session) list and looks up each contrast's file per session, skipping missing ones
— dwi_ap needs no dedicated test-case list.)

Usage: .venv/bin/python 01_03_create_splits_dwi.py
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
BIDS_ROOT  = Path(os.environ["BIDS_ROOT"])
MASKS_ROOT = BIDS_ROOT / "derivatives" / "labels"
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


def discover_dwi_sessions() -> list[dict]:
    """Walk BIDS tree and collect all dwi_ap (dir-AP dwi) sessions that have a
    registered SynthSeg mask."""
    sessions = []
    for sub_dir in sorted(BIDS_ROOT.iterdir()):
        if not sub_dir.name.startswith("sub-"):
            continue
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.name.startswith("ses-"):
                continue
            ses = ses_dir.name
            dwi = ses_dir / "dwi" / f"{sub}_{ses}_dir-AP_dwi.nii.gz"
            mask = MASKS_ROOT / sub / ses / "dwi" / f"{sub}_{ses}_dir-AP_dwi_label-synthseg_dseg.nii.gz"
            if not dwi.exists():
                continue
            if not mask.exists():
                print(f"  WARNING: no SynthSeg mask for {sub} {ses} dwi_ap — skipping")
                continue
            sessions.append({
                "subject":  sub,
                "session":  ses,
                "scanner":  scanner_from_session(ses),
                "dwi_ap":   str(dwi.resolve()),
                "mask":     str(mask.resolve()),
                "case_id":  f"{sub}_{ses}_dwi_ap",
            })
    return sessions


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_sessions = discover_dwi_sessions()
    print(f"Found {len(all_sessions)} dwi_ap sessions with SynthSeg masks")

    all_subjects = sorted({s["subject"] for s in all_sessions})
    print(f"Subjects with usable dwi_ap: {len(all_subjects)} (of 20 total)")

    # ── Test subject selection — IDENTICAL shuffle/seed to 01_01, so subject
    # groups match the T1w/T2w splits exactly. ─────────────────────────────────
    rng = random.Random(SEED)
    # Shuffle must run over the SAME 20-subject universe as 01_01/01_02, not just
    # the subjects that happen to have dwi_ap, so the shuffle order — and thus
    # which subjects land in test/train/fold buckets — is identical across
    # modalities. Discover the full 20-subject list from T1w (always complete).
    t1w_subjects = sorted({
        p.parent.parent.parent.name for p in (BIDS_ROOT).glob("sub-*/ses-*/anat/*_T1w.nii.gz")
    })
    assert len(t1w_subjects) == 20, f"Expected 20 T1w subjects, got {len(t1w_subjects)}"
    shuffled = t1w_subjects.copy()
    rng.shuffle(shuffled)

    test_subjects  = set(shuffled[:N_TEST_SUBJECTS])
    trainval_order = shuffled[N_TEST_SUBJECTS:]

    print(f"\nTest subjects ({N_TEST_SUBJECTS}): {sorted(test_subjects)}")
    print(f"Train/val subjects (of 16 universe): {sorted(trainval_order)}")

    trainval_cases = [
        s for s in all_sessions
        if s["subject"] not in test_subjects and s["scanner"] not in TEST_SCANNERS
    ]
    missing_trainval_subjects = set(trainval_order) - {s["subject"] for s in trainval_cases}
    if missing_trainval_subjects:
        print(f"  NOTE: trainval subjects with zero usable dwi_ap sessions: "
              f"{sorted(missing_trainval_subjects)} (fold sizes will be uneven)")

    print(f"\nTrain/val cases (dwi_ap): {len(trainval_cases)}")

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

    splits_path = SPLITS_DIR / "onharmony_dwi_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWritten: {splits_path}")

    # Persist the full session list too (convert script needs image/mask paths).
    cases_path = SPLITS_DIR / "dwi_trainval_cases.json"
    with open(cases_path, "w") as f:
        json.dump(trainval_cases, f, indent=2)
    print(f"Written: {cases_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
