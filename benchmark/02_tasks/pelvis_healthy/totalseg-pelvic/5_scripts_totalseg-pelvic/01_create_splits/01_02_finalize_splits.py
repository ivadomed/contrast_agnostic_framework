#!/usr/bin/env python3
"""
Phase 2 of totalseg-pelvic's split pipeline (BIDS-era rewrite, 2026-09-15): build the
final 3-fold split from each modality's `usable_cases_verified.json` (written by the
rewritten `02_nnunet/02_00_convert.py`, which converts from the BIDS layer staged by
`00_utils/00_01_bidsify.py`). Run AFTER both bidsify.py and 02_00_convert.py.

Top-up is no longer this script's job -- `00_01_bidsify.py --target-usable N` already
tries candidates until N usable cases are found (or the candidate pool is exhausted), so
by the time this script runs, the usable count should already be at target. This script
just builds the 3-fold split from whatever came out the other end.

Output (4_splits_totalseg-pelvic/{ct,mri}/):
  splits_final.json   3 folds (permanent project policy -- folds 0/1/2 only)
  test_cases.json     sealed held-out cases
  partition.json      human-readable audit trail

Usage:
  .venv/bin/python 01_02_finalize_splits.py
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

RNG_SEED = 42
N_FOLDS = 3
TEST_FRACTION = 0.15

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_ROOT = DATASET_ROOT / "4_splits_totalseg-pelvic"


def _build_split(cases: list[str], rng: random.Random) -> tuple[list[dict], list[str]]:
    cases = sorted(cases)
    rng.shuffle(cases)
    n_test = max(1, round(len(cases) * TEST_FRACTION))
    test_cases = sorted(cases[:n_test])
    trainval = cases[n_test:]
    rng.shuffle(trainval)
    fold_of = {c: i % N_FOLDS for i, c in enumerate(trainval)}
    folds = []
    for k in range(N_FOLDS):
        val = sorted(c for c in trainval if fold_of[c] == k)
        train = sorted(c for c in trainval if fold_of[c] != k)
        folds.append({"train": train, "val": val})
    return folds, test_cases


def _finalize(name: str, n_available_total: int) -> None:
    manifest = json.loads((SPLITS_ROOT / name / "usable_cases_verified.json").read_text())
    usable = manifest["usable"]
    n_rejected = len(manifest.get("rejected", []))

    folds, test_cases = _build_split(usable, random.Random(RNG_SEED))
    out_dir = SPLITS_ROOT / name
    (out_dir / "splits_final.json").write_text(json.dumps(folds, indent=2))
    (out_dir / "test_cases.json").write_text(json.dumps({"test": test_cases}, indent=2))
    (out_dir / "partition.json").write_text(json.dumps({
        "n_available_total": n_available_total,
        "n_bids_rejected": n_rejected,
        "n_usable_final": len(usable),
        "n_test": len(test_cases),
        "n_trainval": len(usable) - len(test_cases),
        "seed": RNG_SEED,
        "test_fraction": TEST_FRACTION,
    }, indent=2))
    n_train0 = len(folds[0]["train"]); n_val0 = len(folds[0]["val"])
    print(f"[finalize][{name}] {len(usable)} usable cases -> {len(test_cases)} test, "
          f"{len(usable) - len(test_cases)} trainval across {N_FOLDS} folds "
          f"(fold0: {n_train0} train / {n_val0} val)")


def main() -> None:
    _finalize("mri", n_available_total=298)
    _finalize("ct", n_available_total=1228)


if __name__ == "__main__":
    main()
