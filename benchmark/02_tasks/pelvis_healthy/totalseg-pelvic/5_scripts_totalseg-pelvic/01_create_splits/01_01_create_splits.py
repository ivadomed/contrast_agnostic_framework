#!/usr/bin/env python3
"""
Phase 1 of totalseg-pelvic's split pipeline: write each modality's CANDIDATE case pool —
NOT the final 3-fold split. Usability (does the case actually have any non-background
pelvic/hip content? see 02_nnunet/02_00_convert.py's corrected docstring) can only be
known after fetching+merging each candidate's 10 label masks, so the final split has to
be built AFTER conversion, by 01_02_finalize_splits.py, from usable_cases.json.

Case lists come directly from each Zenodo archive's own zip central directory (top-level
"sNNNN/" entries) — the same real-counted method used during the pre-flight checklist
(claimed-N vs counted-N: CT 1228/1228, MRI 298/298, both exact) — NOT from meta.csv's
bundled train/test split.

CT CANDIDATE SAMPLING: the public CT release (1228 cases) is far larger than any other
dataset in this project's roster. This script samples CT_SAMPLE_N candidates (seed 42)
from the full 1228 as the FIRST candidate batch. Roughly 1/3 of candidates turn out to
have empty pelvic/hip content (varying scan FOV — see 02_00_convert.py), so the final
usable count from this first batch alone will typically land well below CT_SAMPLE_N;
01_02_finalize_splits.py tops up with additional deterministic candidates (continuing the
same seeded order, no overlap with the first batch) if the usable count after this batch
is short of the target range. MRI uses its full pool (298) as the sole candidate batch —
no top-up mechanism for MRI (matches the original "use MRI in full" intent, just now
correctly interpreted as "all non-empty MRI cases", not literally all 298 files).

Output (4_splits_totalseg-pelvic/{ct,mri}/):
  candidate_cases.json   {"candidates": [...]}  — this phase's output

Usage:
  source ../00_utils/env.sh        # (or env_mri.sh — this script does BOTH regardless)
  .venv/bin/python 01_01_create_splits.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_utils"))
from zenodo_zip import CT_ZENODO_URL, MRI_ZENODO_URL, open_zip  # noqa: E402

RNG_SEED = 42
CT_SAMPLE_N = 200          # first-batch candidate count — see module docstring for why the final usable count differs

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_ROOT = DATASET_ROOT / "4_splits_totalseg-pelvic"

CT_ZIP = Path(os.environ.get("TOTALSEG_CT_ZIP", "/scratch/paulh/totalseg_preflight/ct_full.zip"))
MRI_ZIP = Path(os.environ.get("TOTALSEG_MRI_ZIP", "/scratch/paulh/totalseg_preflight/mri.zip"))


def _case_ids_from_zip(local_path: Path, url: str) -> list[str]:
    with open_zip(local_path, url) as zf:
        names = zf.namelist()
    ids = sorted({n.split("/")[0] for n in names if "/" in n and n.split("/")[0].startswith("s")})
    return ids


def _write_candidates(name: str, all_cases: list[str], sample_n: int | None) -> None:
    rng = random.Random(RNG_SEED)
    pool = sorted(all_cases)
    rng.shuffle(pool)  # deterministic order — top-up (if any) continues from here, see 01_02
    if sample_n is not None and len(pool) > sample_n:
        candidates = sorted(pool[:sample_n])
        print(f"[create_splits][{name}] {len(all_cases)} available -> sampled {len(candidates)} "
              f"first-batch candidates (seed={RNG_SEED})")
    else:
        candidates = sorted(pool)
        print(f"[create_splits][{name}] {len(all_cases)} available -> using all as candidates")

    out_dir = SPLITS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidate_cases.json"
    if path.exists():
        print(f"[create_splits][{name}] candidate_cases.json already exists — leaving it "
              f"untouched (delete it first if you really want to resample from scratch, "
              f"which would orphan any already-converted cases' usability info)")
        return
    path.write_text(json.dumps({"candidates": candidates, "seed": RNG_SEED}, indent=2))
    print(f"[create_splits][{name}] wrote {path}")


def main() -> None:
    print(f"[create_splits] CT source: {'local ' + str(CT_ZIP) if CT_ZIP.exists() else 'remote (HTTP range-read) ' + CT_ZENODO_URL}")
    print(f"[create_splits] MRI source: {'local ' + str(MRI_ZIP) if MRI_ZIP.exists() else 'remote (HTTP range-read) ' + MRI_ZENODO_URL}")

    ct_cases = _case_ids_from_zip(CT_ZIP, CT_ZENODO_URL)
    mri_cases = _case_ids_from_zip(MRI_ZIP, MRI_ZENODO_URL)
    print(f"[create_splits] CT zip: {len(ct_cases)} cases found (expect 1228)")
    print(f"[create_splits] MRI zip: {len(mri_cases)} cases found (expect 298)")

    _write_candidates("ct", ct_cases, CT_SAMPLE_N)
    _write_candidates("mri", mri_cases, None)


if __name__ == "__main__":
    main()
