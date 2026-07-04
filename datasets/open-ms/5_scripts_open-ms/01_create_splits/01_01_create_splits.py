#!/usr/bin/env python3
"""
Create the open-ms train/test partition and 4-fold CV splits — PATIENT-LEVEL, no leakage.

30 patients (open_ms_data cross-sectional). We hold out 8 patients as a dedicated test
set (never seen in any fold) and use the remaining 22 as the training pool for 4-fold CV.
The 8 test patients are chosen STRATIFIED by lesion burden (evenly spaced ranks of total
lesion volume) so the test set spans low→high lesion load. Deterministic (fixed seed).

Writes to 4_splits_open-ms/:
  partition.json     {"train_pool": [...22...], "test": [...8...], "lesion_voxels": {...}}
  splits_final.json  nnUNet 4-fold CV over the 22 train-pool patients (list of {train,val})
  test_cases.json    {"test": [...8...]}  (read by the trainer's contamination guard)

Run:  bash 01_create_splits/01_01_create_splits.sh   (or: python this file)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib

SEED = 1337
N_TEST = 8
N_FOLDS = 4

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/open-ms
RAW = DATASET_ROOT / "0_raw_open-ms"
SPLITS = DATASET_ROOT / "4_splits_open-ms"


def main() -> None:
    patients = sorted(p.name for p in RAW.glob("patient*") if p.is_dir())
    if len(patients) != 30:
        raise SystemExit(f"Expected 30 patients in {RAW}, found {len(patients)}")

    # Lesion burden per patient (for stratified test selection + later size analysis).
    lesion_vox = {}
    for pid in patients:
        gt = nib.load(str(RAW / pid / "consensus_gt.nii.gz"))
        lesion_vox[pid] = int((np.asanyarray(gt.dataobj) > 0).sum())

    # Stratified test set: rank by lesion volume, take N_TEST evenly-spaced ranks.
    by_burden = sorted(patients, key=lambda p: lesion_vox[p])
    test_idx = np.linspace(0, len(by_burden) - 1, N_TEST).round().astype(int)
    test = sorted({by_burden[i] for i in test_idx})
    # linspace can collide on ties at the ends — top up deterministically if <N_TEST.
    i = 0
    while len(test) < N_TEST and i < len(by_burden):
        test = sorted(set(test) | {by_burden[i]}); i += 1
    train_pool = sorted(p for p in patients if p not in set(test))
    assert len(test) == N_TEST and len(train_pool) == 30 - N_TEST
    assert not (set(test) & set(train_pool))

    # 4-fold CV over the train pool (seeded patient-level shuffle).
    rng = np.random.default_rng(SEED)
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    folds = [shuffled[k::N_FOLDS] for k in range(N_FOLDS)]   # round-robin → balanced sizes
    splits = []
    for k in range(N_FOLDS):
        val = sorted(folds[k])
        train = sorted(p for p in train_pool if p not in set(val))
        assert not (set(train) & set(val))
        splits.append({"train": train, "val": val})

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps(
        {"train_pool": train_pool, "test": test, "lesion_voxels": lesion_vox,
         "seed": SEED}, indent=2))
    (SPLITS / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({"test": test}, indent=2))

    print(f"Train pool ({len(train_pool)}): {train_pool}")
    print(f"Held-out test ({len(test)}, stratified by lesion burden): {test}")
    for k, s in enumerate(splits):
        print(f"  fold {k}: {len(s['train'])} train / {len(s['val'])} val   val={s['val']}")
    print(f"  test lesion voxels: {[lesion_vox[p] for p in test]}")
    print(f"→ {SPLITS}/(partition|splits_final|test_cases).json")


if __name__ == "__main__":
    main()
