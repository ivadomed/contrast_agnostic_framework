#!/usr/bin/env python3
"""
Import the TUM collaborators' externally computed spine results (TSV archive)
into this project's standard metrics layout:

  8_results_healthy-spine-tum/02_metrics/healthy_spine_tum_model/<train>/
      <category>_healthy-spine-tum_<train>_<method>_<RUN_TS>/fold{k}/eval_all.csv

Spine was trained and predicted on the collaborators' pipeline, not ours; this
script only converts their per-subject metrics. Each TSV row is one subject;
columns are "<label>-<metric>" for label in {vertebra, ivd, spinal_canal}:
  dice <- <label>-global_bin_dsc   (whole-structure binary Dice)
  hd95 <- <label>-global_bin_hd95  (its whole-structure counterpart; blank or
                                    non-finite -> the literal "nan", exactly
                                    what evaluate.py writes for an empty mask;
                                    an empty cell crashes the shared loaders)

METHOD MAPPING IS BY VERIFIED CONTENT, NOT FOLDER NAME. The 2026-10-02 drop
renamed folders while adding HD95, and the names are not self-explanatory:
  ImageContrastV26-05+Paper  == PALETTE-05+Paper (Ours): Dice identical on all
                                4536 rows per training contrast, plus HD95
  newd                       == Paper (Auglab): Dice identical on all rows, plus HD95
The old-name folders for those two carry no HD95, so the new names win. Re-run
--verify after any new drop before trusting a mapping.

The 2026-08-05 import of Synthseg10NoEM+GE was TRUNCATED (8-81 subjects per
test group instead of 27-219); the 2026-10-02 drop is complete and matches every
other method's subject set exactly.

Usage:
  .venv/bin/python 06_00_import_tum_tsv.py <extracted tsv/ dir> [--verify]
  (extract with Python zipfile, not unzip: the archive stores its directories
   with mode 000, which unzip reproduces as unreadable folders)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
DST = DATASET_ROOT / "8_results_healthy-spine-tum/02_metrics/healthy_spine_tum_model"
# Kept from the first import so every config's run ids stay valid; it is an
# import stamp, not a training timestamp (this repo never trained spine).
RUN_TS = "20260805_130846"

TRAIN = {"SG_CT": "ct", "SG_in-phase": "inphase"}
# source folder -> (category, method slug)
METHODS = {
    "Base":                      ("nnUNet", "baseline"),
    "newd":                      ("auglab", "auglab_default"),
    "Synthseg10NoEM+GE":         ("auglab", "synthseg_noEM"),
    "Synthseg10+GE":             ("auglab", "synthseg_EM"),
    "ImageContrastV26-05+Paper": ("auglab", "auglabAug_v26_6_2"),   # OURS
    "PALETTE-05+GE":             ("auglab", "palette05_ge"),        # extra arm, not in the suite
}
# (new folder, folder whose Dice it must reproduce exactly)
EQUIVALENT = [("ImageContrastV26-05+Paper", "PALETTE-05+Paper"), ("newd", "Paper")]
LABELS = ["vertebra", "ivd", "spinal_canal"]
FNAME = re.compile(r"^(spider|spinegan)-test-(.+)\.tsv$")


def group_of(name: str) -> str:
    m = FNAME.match(name)
    if not m:
        raise SystemExit(f"unexpected file name {name!r}")
    return f"{m[1]}_{m[2].replace('dixon_part-', 'dixon_')}"


def finite_or_nan(v: str) -> str:
    try:
        return v if math.isfinite(float(v)) else "nan"
    except ValueError:
        return "nan"


def rows_of(folder: Path):
    for fold_dir in sorted(folder.glob("fold-*")):
        k = fold_dir.name.split("-")[1]
        for tsv in sorted(fold_dir.glob("*.tsv")):
            g = group_of(tsv.name)
            with open(tsv, newline="") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    for lab in LABELS:
                        yield (k, g, r["subject_name"], lab,
                               r[f"{lab}-global_bin_dsc"],
                               finite_or_nan(r.get(f"{lab}-global_bin_hd95", "")))


def verify(src: Path) -> None:
    for tdir in TRAIN:
        for new, old in EQUIVALENT:
            a = {r[:4]: r[4] for r in rows_of(src / tdir / new)}
            b = {r[:4]: r[4] for r in rows_of(src / tdir / old)}
            bad = sum(1 for k in a if b.get(k) != a[k])
            if bad or set(a) != set(b):
                raise SystemExit(f"{tdir}: {new} does not reproduce {old}'s Dice "
                                 f"({bad} mismatches) -- mapping is wrong for this drop")
            print(f"verified {tdir}: {new} == {old} on all {len(a)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="extracted tsv/ directory")
    ap.add_argument("--verify", action="store_true", help="only check the mapping")
    a = ap.parse_args()
    verify(a.src)
    if a.verify:
        return
    n = 0
    for tdir, train in TRAIN.items():
        for folder, (cat, slug) in METHODS.items():
            src = a.src / tdir / folder
            if not src.is_dir():
                raise SystemExit(f"missing source folder {src}")
            by_fold: dict = {}
            for k, g, case, lab, d, h in rows_of(src):
                by_fold.setdefault(k, []).append((g, case, lab, d, h))
            for k, rows in by_fold.items():
                out = DST / train / f"{cat}_healthy-spine-tum_{train}_{slug}_{RUN_TS}" / f"fold{k}"
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "eval_all.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["group", "case", "label", "dice", "hd95"])
                    w.writerows(rows)
                n += len(rows)
    print(f"wrote {n} rows under {DST}")


if __name__ == "__main__":
    sys.exit(main())
