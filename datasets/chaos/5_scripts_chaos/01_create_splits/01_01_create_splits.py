#!/usr/bin/env python3
"""
Deterministic splits for CHAOS.

The sealed official test set has no public GT, so ALL splits come from the public
training half (20 MR + 20 CT patients, each with ground truth).

Strategy
--------
- Training pool = MR T1-DUAL in-phase only (single-channel, 4 organ labels).
- Hold out N_HOLDOUT_MR MR patients as a sealed internal test set; they are
  evaluated on ALL MR modalities (in-phase / out-phase / T2-SPIR).
- Remaining MR patients → N_FOLDS-fold subject-level CV (these are the only cases
  put into nnUNet imagesTr).
- CT is NEVER trained on (disjoint patients, liver-only) → all 20 CT patients are a
  pure cross-modality liver test set.
- Seeded (SEED) → fully deterministic.

Outputs (4_splits_chaos/)
-------
    splits_final.json   -- N_FOLDS-fold CV over the MR-CV pool, nnUNet format
    test_cases.json     -- internal test sets + per-modality / scoreable-organ metadata

Also copies splits_final.json into the nnUNet preprocessed dataset dir.

Run AFTER 00_00_download_and_bidsify.py (needs participants.tsv).
    .venv/bin/python 01_01_create_splits.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from splits_lib import kfold_splits, write_splits_final      # noqa: E402
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_chaos" / "chaos-abdominal"
NNUNET_PRE   = DATASET_ROOT / "2_nnUNet_chaos" / "preprocessed"
SPLITS_DIR   = DATASET_ROOT / "4_splits_chaos"

NNUNET_DATASET = "Dataset060_CHAOS_MR_T1in"
N_FOLDS        = 4
N_HOLDOUT_MR   = 4
SEED           = 12345

# Modalities each internal-test group is evaluated on (→ imagesTs_<tag> dirs).
MR_TEST_MODALITIES = ["t1in", "t1out", "t2spir"]
CT_TEST_MODALITIES = ["ct"]
# Scoreable organ labels per source (CT is liver-only).
SCOREABLE = {"MR": [1, 2, 3, 4], "CT": [1]}


def discover_cases() -> tuple[list[str], list[str]]:
    tsv = BIDS_ROOT / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")
    lines = tsv.read_text().strip().split("\n")
    header = lines[0].split("\t")
    mi, pi = header.index("modality"), header.index("participant_id")
    mr, ct = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        case = cols[pi].removeprefix("sub-")
        (mr if cols[mi] == "MR" else ct).append(case)
    return sorted(mr), sorted(ct)


def main() -> None:
    mr_cases, ct_cases = discover_cases()
    print(f"Found {len(mr_cases)} MR + {len(ct_cases)} CT cases (public/labelled).")

    rng = random.Random(SEED)
    shuffled = mr_cases.copy()
    rng.shuffle(shuffled)

    mr_test = sorted(shuffled[:N_HOLDOUT_MR])
    mr_cv   = shuffled[N_HOLDOUT_MR:]
    print(f"MR internal test: {len(mr_test)} → {mr_test}")
    print(f"MR CV pool      : {len(mr_cv)} → {N_FOLDS}-fold CV")
    print(f"CT liver test   : {len(ct_cases)} (all CT, never trained)")

    # N_FOLDS-fold subject-level CV over the MR-CV pool (shared chunking; see splits_lib).
    splits = kfold_splits(mr_cv, N_FOLDS)
    for k, s in enumerate(splits):
        print(f"  Fold {k}: {len(s['train'])} train, {len(s['val'])} val")

    # Sanity: no train/val overlap, no holdout contamination.
    for k, s in enumerate(splits):
        assert not (set(s["train"]) & set(s["val"])), f"fold {k} train/val overlap"
        assert not (set(s["train"] + s["val"]) & set(mr_test)), f"fold {k} holdout leak"
    print("Sanity checks passed (no overlap, no holdout contamination).")

    # Write splits_final.json to 4_splits/ and copy into the nnUNet preprocessed dir.
    splits_path = write_splits_final(splits, SPLITS_DIR, NNUNET_PRE, NNUNET_DATASET)
    print(f"Written: {splits_path}  (+ copied into {NNUNET_PRE / NNUNET_DATASET})")

    test_meta = {
        "mr_internal_test": mr_test,
        "ct_test": sorted(ct_cases),
        "mr_test_modalities": MR_TEST_MODALITIES,
        "ct_test_modalities": CT_TEST_MODALITIES,
        "scoreable_organs": SCOREABLE,
    }
    test_path = SPLITS_DIR / "test_cases.json"
    test_path.write_text(json.dumps(test_meta, indent=2))
    print(f"Written: {test_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
