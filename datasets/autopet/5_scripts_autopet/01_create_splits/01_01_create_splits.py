#!/usr/bin/env python3
"""
Build this project's OWN patient-level split for autopet, from the real per-case
positivity/orientation manifest produced by 01_00_scan_labels.py (run first, via
01_00_scan_labels.sh) — NOT the archive's reference splits_final.json (challenge's own
5-fold CV over ALL data, positives+negatives mixed, no sealed test), and NOT the
"diagnosis" metadata column (verified 2026-09-13: patient-level disease category, not
per-study lesion presence — 30 FDG patients have differing diagnosis across their own
studies, e.g. one study LYMPHOMA-positive, a later surveillance study genuinely NEGATIVE;
psma_metadata.csv has no diagnosis column at all). Real usable-N comes only from checking
each study's own label file for a nonzero voxel.

CLEAN CASE IDs — the archive's own FDG case ids embed a free-text TCIA study description
with spaces (e.g. "fdg_01140d52d8_08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839")
— fragile for any shell script building paths from them. This script derives a clean,
compact id (fdg_<hex10>_<YYYYMMDD>) for every FDG case and writes case_id_map.json
(clean_id -> original archive case string) so 02_00_convert.py / 05_00b_*.py can still
locate the right zip members. PSMA case ids are already clean natively
(psma_<hex16>_<YYYY-MM-DD>) — normalized here to psma_<hex16>_<YYYYMMDD> for consistency,
same mapping mechanism reused rather than special-cased.

Splits BY PATIENT (hex id), not by study — some patients have multiple studies/timepoints
(30 FDG patients confirmed above), and splitting by study would leak a patient's second
study across train/val/test.

Output (4_splits_autopet/):
  case_id_map.json    clean_id -> original archive case id   (ALL cases, both tracers)
  splits_final.json   3 folds (permanent project policy), FDG POSITIVE cases only
  test_cases.json     sealed held-out FDG positive cases
  partition.json       human-readable audit trail

Usage:
  source ../00_utils/env.sh
  .venv/bin/python 01_01_create_splits.py
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

RNG_SEED = 42
N_FOLDS = 3                # permanent project fold policy — folds 0/1/2 only
TEST_FRACTION = 0.15       # sealed held-out slice, carved out before fold-splitting

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_autopet")))

_FDG_DATE_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})-")


def _clean_fdg_id(messy_case_id: str) -> str:
    # messy_case_id looks like "fdg_<hex10>_<MM-DD-YYYY>-NA-<free text...>"
    parts = messy_case_id.split("_", 2)
    assert parts[0] == "fdg", f"unexpected prefix in {messy_case_id!r}"
    hex_id = parts[1]
    rest = parts[2]
    m = _FDG_DATE_RE.match(rest)
    if not m:
        raise ValueError(f"could not parse date out of {messy_case_id!r} (rest={rest!r})")
    mm, dd, yyyy = m.groups()
    return f"fdg_{hex_id}_{yyyy}{mm}{dd}"


def _clean_psma_id(native_case_id: str) -> str:
    # native_case_id looks like "psma_<hex16>_<YYYY-MM-DD>" — already clean, just
    # collapse the date to YYYYMMDD for consistency with the FDG side.
    parts = native_case_id.split("_")
    assert parts[0] == "psma", f"unexpected prefix in {native_case_id!r}"
    hex_id = parts[1]
    date = parts[2].replace("-", "")
    return f"psma_{hex_id}_{date}"


def _patient_id(clean_id: str) -> str:
    # "fdg_<hex>_<date>" -> "fdg_<hex>" (patient-level grouping key)
    tracer, hex_id, _date = clean_id.split("_")
    return f"{tracer}_{hex_id}"


def main() -> None:
    manifest_path = SPLITS_DIR / "label_scan_manifest.json"
    scan = json.loads(manifest_path.read_text())
    manifest = scan["manifest"]
    print(f"[create_splits] loaded manifest: {len(manifest)} cases "
          f"(FDG {scan['n_total']['fdg']}, PSMA {scan['n_total']['psma']})")
    print(f"[create_splits] positive per archive scan: FDG {scan['n_positive']['fdg']}, "
          f"PSMA {scan['n_positive']['psma']}")
    print(f"[create_splits] axcode distribution: {scan['axcode_counts']}")
    non_standard_ax = {k: v for k, v in scan["axcode_counts"].items() if k != "L_A_S" and k != "L_P_S"}
    if non_standard_ax:
        print(f"[create_splits] ⚠️ non-uniform orientations found: {non_standard_ax} — "
              f"verify preprocessing reorients consistently, do not assume LAS for all cases")

    case_id_map: dict[str, str] = {}
    entries_by_clean_id: dict[str, dict] = {}
    for m in manifest:
        if m["tracer"] == "fdg":
            clean_id = _clean_fdg_id(m["case_id"])
        else:
            clean_id = _clean_psma_id(m["case_id"])
        if clean_id in case_id_map:
            raise ValueError(
                f"clean id collision: {clean_id!r} maps to both {case_id_map[clean_id]!r} "
                f"and {m['case_id']!r} — two studies same patient same day? "
                "extend the clean-id scheme (e.g. append an index) before proceeding."
            )
        case_id_map[clean_id] = m["case_id"]
        entries_by_clean_id[clean_id] = m

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "case_id_map.json").write_text(json.dumps(case_id_map, indent=2, sort_keys=True))
    print(f"[create_splits] wrote case_id_map.json ({len(case_id_map)} entries)")

    # FDG-only, positive-only pool for the actual training splits (PSMA never trains —
    # see 00_utils/env.sh; PSMA cases are consumed directly by
    # 05_00b_build_psma_test_inputs.py via case_id_map, not through splits_final.json).
    fdg_positive_clean_ids = sorted(
        cid for cid, e in entries_by_clean_id.items()
        if e["tracer"] == "fdg" and e["n_positive_voxels"] > 0
    )
    n_fdg_pos, n_fdg_total = scan["n_positive"]["fdg"], scan["n_total"]["fdg"]
    print(f"[create_splits] FDG positive studies: {n_fdg_pos}/{n_fdg_total} "
          f"(expected ~501-600ish per the paper's patient-level figure — this is the "
          f"STUDY-level count, patients with multiple studies can contribute >1)")

    patient_to_cases: dict[str, list[str]] = {}
    for cid in fdg_positive_clean_ids:
        patient_to_cases.setdefault(_patient_id(cid), []).append(cid)

    patients = sorted(patient_to_cases.keys())
    rng = random.Random(RNG_SEED)
    rng.shuffle(patients)

    n_test = max(1, round(len(patients) * TEST_FRACTION))
    test_patients = sorted(patients[:n_test])
    trainval_patients = sorted(patients[n_test:])

    rng.shuffle(trainval_patients)
    fold_of_patient = {p: i % N_FOLDS for i, p in enumerate(trainval_patients)}

    def cases_for(pats: list[str]) -> list[str]:
        out: list[str] = []
        for p in pats:
            out.extend(patient_to_cases[p])
        return sorted(out)

    folds = []
    for k in range(N_FOLDS):
        val_p = [p for p in trainval_patients if fold_of_patient[p] == k]
        train_p = [p for p in trainval_patients if fold_of_patient[p] != k]
        folds.append({"train": cases_for(train_p), "val": cases_for(val_p)})

    test_cases = cases_for(test_patients)

    (SPLITS_DIR / "splits_final.json").write_text(json.dumps(folds, indent=2))
    (SPLITS_DIR / "test_cases.json").write_text(json.dumps({"test": test_cases}, indent=2))
    (SPLITS_DIR / "partition.json").write_text(json.dumps({
        "n_fdg_positive_patients": len(patients),
        "n_fdg_positive_studies": len(fdg_positive_clean_ids),
        "test_patients": test_patients,
        "trainval_patients": trainval_patients,
        "fold_of_patient": fold_of_patient,
        "seed": RNG_SEED,
        "test_fraction": TEST_FRACTION,
    }, indent=2))

    print(f"[create_splits] {len(patients)} positive FDG patients "
          f"({len(fdg_positive_clean_ids)} studies) -> "
          f"{len(test_patients)} test patients ({len(test_cases)} cases), "
          f"{len(trainval_patients)} trainval patients across {N_FOLDS} folds")
    print(f"[create_splits] wrote splits_final.json, test_cases.json, partition.json")


if __name__ == "__main__":
    main()
