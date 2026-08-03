#!/usr/bin/env python3
"""
Create the picai-prostate train/test partition and 4-fold CV splits — PATIENT-LEVEL,
no leakage.

Grouping is by PATIENT, not by study: a handful of PI-CAI patients contribute two studies
(case id is "<patient>_<study>"), and both must land on the same side of every boundary or
the same prostate leaks between train and test.

Held-out test set = TEST_FRAC of patients, chosen STRATIFIED by lesion burden (evenly
spaced ranks of total lesion volume, as open-ms does) so the test set spans small→large
lesions rather than accidentally collecting only the easy big ones. The remaining patients
form the training pool for 4-fold CV.

FOLD COUNT: 4 folds are written but only folds 0/1/2 are ever trained (CLAUDE.md "FOLD
POLICY" — 3 folds only, permanently). The file has 4 so it matches every other dataset in
the suite and so nnUNetTrainerPICAIProstateBase's N_EXPECTED_FOLDS guard is the same
constant everywhere.

Writes to 4_splits_picai-prostate/:
  partition.json     {"train_pool": [...], "test": [...], "lesion_voxels": {...}, ...}
  splits_final.json  nnUNet 4-fold CV over the train-pool CASES (list of {train,val})
  test_cases.json    {"test": [...case ids...]}  (read by the trainer's contamination guard)

Run:  bash 01_create_splits/01_01_create_splits.sh
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SEED = 1337
TEST_FRAC = 0.25
N_FOLDS = 4

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/picai-prostate


def main() -> None:
    bids_root = Path(os.environ.get(
        "BIDS_ROOT", DATASET_ROOT / "1_BIDS_picai-prostate" / "picai-prostate-bpmri"))
    splits_dir = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_picai-prostate"))

    manifest = json.loads((bids_root / "cases.json").read_text())
    if not manifest:
        raise SystemExit(f"cases.json at {bids_root} is empty — run 00_01_bidsify.sh first")

    # patient -> [case ids], and patient -> total lesion burden across its studies
    by_patient: dict[str, list[str]] = {}
    burden: dict[str, int] = {}
    for cid, meta in manifest.items():
        pid = meta["patient"]
        by_patient.setdefault(pid, []).append(cid)
        burden[pid] = burden.get(pid, 0) + int(meta["n_lesion_voxels"])
    patients = sorted(by_patient)

    n_test = max(1, round(len(patients) * TEST_FRAC))
    ranked = sorted(patients, key=lambda p: (burden[p], p))
    idx = np.linspace(0, len(ranked) - 1, n_test).round().astype(int)
    test_pat = sorted({ranked[i] for i in idx})
    # linspace can collide on ties — top up deterministically until we have n_test.
    i = 0
    while len(test_pat) < n_test and i < len(ranked):
        test_pat = sorted(set(test_pat) | {ranked[i]})
        i += 1
    train_pat = sorted(p for p in patients if p not in set(test_pat))
    assert not (set(test_pat) & set(train_pat))

    test_cases = sorted(c for p in test_pat for c in by_patient[p])
    pool_cases = sorted(c for p in train_pat for c in by_patient[p])

    # 4-fold CV, PATIENT-level round-robin so a patient's studies stay together.
    rng = np.random.default_rng(SEED)
    shuffled = list(train_pat)
    rng.shuffle(shuffled)
    fold_pat = [shuffled[k::N_FOLDS] for k in range(N_FOLDS)]
    splits = []
    for k in range(N_FOLDS):
        val_p = set(fold_pat[k])
        val = sorted(c for p in val_p for c in by_patient[p])
        train = sorted(c for p in train_pat if p not in val_p for c in by_patient[p])
        assert not (set(train) & set(val))
        assert not (set(train + val) & set(test_cases))
        splits.append({"train": train, "val": val})

    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "partition.json").write_text(json.dumps({
        "seed": SEED, "test_frac": TEST_FRAC, "n_folds": N_FOLDS,
        "train_pool_patients": train_pat, "test_patients": test_pat,
        "train_pool": pool_cases, "test": test_cases,
        "lesion_voxels_by_patient": burden,
    }, indent=2))
    (splits_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (splits_dir / "test_cases.json").write_text(json.dumps({"test": test_cases}, indent=2))

    print(f"patients: {len(patients)}  studies: {len(manifest)}")
    print(f"train pool: {len(train_pat)} patients / {len(pool_cases)} studies")
    print(f"held-out test: {len(test_pat)} patients / {len(test_cases)} studies "
          f"(stratified by lesion burden)")
    for k, s in enumerate(splits):
        print(f"  fold {k}: {len(s['train'])} train / {len(s['val'])} val")
    print(f"→ {splits_dir}/(partition|splits_final|test_cases).json")


if __name__ == "__main__":
    main()
