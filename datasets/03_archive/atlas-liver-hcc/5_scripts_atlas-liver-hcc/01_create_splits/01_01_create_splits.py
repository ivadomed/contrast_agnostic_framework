#!/usr/bin/env python3
"""
Create the atlas-liver-hcc train/test partition and 4-fold CV splits — PATIENT-LEVEL,
no leakage. Same shape as open-ms's 01_01_create_splits.py.

60 patients (ATLAS challenge train set, Quinton et al. 2023). We hold out 12 patients
as a dedicated test set (never seen in any fold) and use the remaining 48 as the
training pool for 4-fold CV. The 12 test patients are chosen STRATIFIED by TUMOUR
VOXEL COUNT (label 2, `tumour` — the intra-tissue lesion this dataset actually scores;
`liver`/label 1 is the surrounding-organ boundary, not the stratification target) so
the test set spans low->high tumour burden. Deterministic (fixed seed).

Case ids are assigned as "atlas_XXX" (XXX = zero-padded original ATLAS patient index,
im{XXX}.nii.gz / lb{XXX}.nii.gz in 0_raw) — this is the SAME id 02_nnunet/02_00_convert.py
uses when writing imagesTr/labelsTr, so splits_final.json / test_cases.json need no
translation layer at train time.

Writes to 4_splits_atlas-liver-hcc/:
  partition.json     {"train_pool": [...48...], "test": [...12...], "tumour_voxels": {...}}
  splits_final.json  nnUNet 4-fold CV over the 48 train-pool cases (list of {train,val})
  test_cases.json    {"test": [...12...]}  (read by the trainer's contamination guard)

Run:  bash 01_create_splits/01_01_create_splits.sh   (or: python this file)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib

SEED = 1337
N_PATIENTS = 60
N_TEST = 12
N_FOLDS = 4
TUMOUR_LABEL = 2

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/atlas-liver-hcc
RAW = DATASET_ROOT / "0_raw_atlas-liver-hcc" / "train"
SPLITS = DATASET_ROOT / "4_splits_atlas-liver-hcc"


def case_id(pid: int) -> str:
    return f"atlas_{pid:03d}"


def main() -> None:
    label_files = sorted(RAW.glob("labelsTr/lb*.nii.gz"),
                          key=lambda p: int(p.stem.split(".")[0].removeprefix("lb")))
    if len(label_files) != N_PATIENTS:
        raise SystemExit(f"Expected {N_PATIENTS} patients in {RAW}/labelsTr, found {len(label_files)}")

    # Tumour burden per patient (for stratified test selection + later size analysis).
    tumour_vox: dict[str, int] = {}
    for lf in label_files:
        pid = int(lf.stem.split(".")[0].removeprefix("lb"))
        arr = np.asanyarray(nib.load(str(lf)).dataobj)
        tumour_vox[case_id(pid)] = int((arr == TUMOUR_LABEL).sum())

    cases = sorted(tumour_vox.keys())

    # Stratified test set: rank by tumour volume, take N_TEST evenly-spaced ranks.
    by_burden = sorted(cases, key=lambda c: tumour_vox[c])
    test_idx = np.linspace(0, len(by_burden) - 1, N_TEST).round().astype(int)
    test = sorted({by_burden[i] for i in test_idx})
    # linspace can collide on ties at the ends — top up deterministically if <N_TEST.
    i = 0
    while len(test) < N_TEST and i < len(by_burden):
        test = sorted(set(test) | {by_burden[i]}); i += 1
    train_pool = sorted(c for c in cases if c not in set(test))
    assert len(test) == N_TEST and len(train_pool) == N_PATIENTS - N_TEST
    assert not (set(test) & set(train_pool))

    # 4-fold CV over the train pool (seeded patient-level shuffle).
    rng = np.random.default_rng(SEED)
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    folds = [shuffled[k::N_FOLDS] for k in range(N_FOLDS)]   # round-robin -> balanced sizes
    splits = []
    for k in range(N_FOLDS):
        val = sorted(folds[k])
        train = sorted(c for c in train_pool if c not in set(val))
        assert not (set(train) & set(val))
        splits.append({"train": train, "val": val})

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps(
        {"train_pool": train_pool, "test": test, "tumour_voxels": tumour_vox,
         "seed": SEED}, indent=2))
    (SPLITS / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({"test": test}, indent=2))

    print(f"Train pool ({len(train_pool)}): {train_pool}")
    print(f"Held-out test ({len(test)}, stratified by tumour burden): {test}")
    for k, s in enumerate(splits):
        print(f"  fold {k}: {len(s['train'])} train / {len(s['val'])} val   val={s['val']}")
    print(f"  test tumour voxels: {[tumour_vox[c] for c in test]}")
    print(f"-> {SPLITS}/(partition|splits_final|test_cases).json")


if __name__ == "__main__":
    main()
