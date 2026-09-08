#!/usr/bin/env python3
"""
Per-SOURCE-ID voxel histogram over the whole ToothFairy2 release.

Label-only pass: reads just labelsTr/*.mha straight out of the release zip, skips
the images entirely and does no resampling, so it costs a small fraction of the full
conversion.

WHY THIS EXISTS
---------------
The reduced task (00_utils/toothfairy2_labels.py) merges 32 tooth ids into one
`teeth` class. That merge HIDES the question that matters most for this release:
ToothFairy2 ships two cohorts with different annotation completeness —
`ToothFairy2F_*` (63 volumes, near-complete) and `ToothFairy2P_*` (417 volumes),
and the P cohort annotates maxilla in only 124/417 and maxillary sinus in 4/417.
A per-CLASS presence check on the merged labels cannot tell whether P also omits
the UPPER teeth (ids 11-28) while keeping the lower ones (31-48) — which would mean
the single `teeth` class silently means two different things in the two cohorts,
inside one training set. That is exactly the annotation-style mismatch this project
has been bitten by before, and no downstream metric would reveal it.

So: count every source id separately, per cohort, and write the full histogram to
JSON. Every future "is class X consistently annotated here?" question is then
answerable from this file instead of another pass over 480 volumes.

Run via 00_01_audit_source_labels.sh. Output: <BIDS_ROOT>/source_label_audit.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toothfairy2_labels import (  # noqa: E402
    LOWER_TEETH_IDS, UPPER_TEETH_IDS, LABELS_SOURCE, ALL_TEETH_IDS,
)

TF2_ZIP = Path(os.environ["TF2_ZIP"])
OUT_DIR = Path(os.environ["BIDS_ROOT"])
N_WORKERS = int(os.environ.get("TF2_WORKERS", "16"))
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "tf2_labaudit"

_ZIP: zipfile.ZipFile | None = None
_PREFIX = ""


def _zip() -> zipfile.ZipFile:
    global _ZIP, _PREFIX
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(TF2_ZIP)
        for n in _ZIP.namelist():
            if "labelsTr/" in n:
                _PREFIX = n[: n.index("labelsTr/")]
                break
    return _ZIP


def audit_one(case: str) -> dict:
    z = _zip()
    stage = TMPDIR / case
    stage.mkdir(parents=True, exist_ok=True)
    try:
        member = f"{_PREFIX}labelsTr/{case}.mha"
        out = stage / f"{case}.mha"
        with z.open(member) as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=8 << 20)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    ids, counts = np.unique(arr, return_counts=True)
    return {"case": case,
            "cohort": "F" if "F_" in case else "P",
            "hist": {int(i): int(c) for i, c in zip(ids, counts) if int(i) != 0}}


def main() -> None:
    with zipfile.ZipFile(TF2_ZIP) as z:
        pre = next(n[: n.index("labelsTr/")] for n in z.namelist() if "labelsTr/" in n)
        cases = sorted(Path(n).stem for n in z.namelist()
                       if n.startswith(f"{pre}labelsTr/") and n.endswith(".mha"))
    print(f"{len(cases)} label volumes", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(audit_one, c) for c in cases]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0:
                print(f"  {i}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: r["case"])

    # presence counts per source id, per cohort
    present = defaultdict(Counter)
    n_cohort = Counter(r["cohort"] for r in rows)
    for r in rows:
        for i in r["hist"]:
            present[r["cohort"]][i] += 1

    # THE question this script exists to answer.
    grp = {}
    for coh in sorted(n_cohort):
        def any_of(ids):
            return sum(1 for r in rows if r["cohort"] == coh and any(i in r["hist"] for i in ids))
        grp[coh] = {"n": n_cohort[coh],
                    "upper_teeth": any_of(UPPER_TEETH_IDS),
                    "lower_teeth": any_of(LOWER_TEETH_IDS),
                    "any_teeth": any_of(ALL_TEETH_IDS)}

    name_of = {v: k for k, v in LABELS_SOURCE.items()}
    out = {"n_cases": len(rows), "cohort_sizes": dict(n_cohort),
           "teeth_by_arch": grp,
           "present_by_cohort": {c: {str(i): present[c][i] for i in sorted(present[c])}
                                 for c in sorted(present)},
           "id_names": {str(k): v for k, v in name_of.items()},
           "cases": rows}
    p = OUT_DIR / "source_label_audit.json"
    p.write_text(json.dumps(out, indent=2))

    print("\n=== teeth annotation by dental arch, per cohort ===")
    for coh, g in grp.items():
        print(f"  {coh}: n={g['n']:4d}  upper={g['upper_teeth']:4d}  "
              f"lower={g['lower_teeth']:4d}  any={g['any_teeth']:4d}")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
