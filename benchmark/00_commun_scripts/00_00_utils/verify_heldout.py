#!/usr/bin/env python3
"""Prove that a TRAINING task's held-out test set is completely held out, across ALL of its per-contrast nnU-Net datasets.
Run for every new training dataset after conversion + preprocessing (CPU job), and again after any re-split / re-conversion.

Contract it checks (one subject = one case id, the same id in every contrast's dataset; no contrast suffix):
  1. every dataset has the same TRAIN id set, and the same TEST id set in every imagesTs_<item>/ and labelsTs_<item>/;
  2. TEST ∩ TRAIN = ∅ in every dataset (so a subject can never be a test case of one contrast and a training case of another);
  3. splits_final.json (3 folds): train ∩ val = ∅ per fold, the val folds PARTITION the train pool, nothing outside the pool appears,
     no test id appears in any fold, and the splits file is byte-identical across the contrast datasets;
  4. test_cases.json (if present in --splits-dir) equals the test set;
  5. nnU-Net's preprocessed output (nnUNetPlans_3d_fullres, gt_segmentations) contains ONLY train ids (no test case was preprocessed/fingerprinted);
  6. (optional --participants) the participants.tsv subject ids behind the case ids are unique (one case per subject).
Exit code 1 on any violation; prints a one-line verdict per check.

  .venv/bin/python verify_heldout.py --raw <2_nnUNet_x/raw> --preprocessed <2_nnUNet_x/preprocessed> --splits-dir <4_splits_x> \
      [--datasets Dataset140_X Dataset141_X]   # default: every Dataset*/ under --raw
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path


def ids(d: Path, strip_channel: bool) -> set:
    out = set()
    for f in d.glob("*.nii.gz"):
        n = f.name[:-7]
        out.add(re.sub(r"_\d{4}$", "", n) if strip_channel else n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--preprocessed", required=True); ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--datasets", nargs="*"); ap.add_argument("--participants")
    a = ap.parse_args()
    raw, prep = Path(a.raw), Path(a.preprocessed)
    dsets = a.datasets or sorted(p.name for p in raw.glob("Dataset*") if p.is_dir())
    bad = []
    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
        if not ok: bad.append(name)

    train, test, splits = {}, {}, {}
    for d in dsets:
        r = raw / d
        train[d] = ids(r / "imagesTr", True)
        check(f"{d}: imagesTr ids == labelsTr ids", train[d] == ids(r / "labelsTr", False))
        items = sorted(p.name[len("imagesTs_"):] for p in r.glob("imagesTs_*"))
        per = {}
        for it in items:
            per[it] = ids(r / f"imagesTs_{it}", True)
            check(f"{d}: imagesTs_{it} ids == labelsTs_{it} ids", per[it] == ids(r / f"labelsTs_{it}", False))
        test[d] = set.union(*per.values()) if per else set()
        check(f"{d}: identical test ids in every item {items}", all(v == test[d] for v in per.values()), f"{ {k: len(v) for k, v in per.items()} }")
        check(f"{d}: TEST ∩ TRAIN = ∅", not (train[d] & test[d]), str(sorted(train[d] & test[d])[:5]))
        sp = prep / d / "splits_final.json"
        check(f"{d}: splits_final.json present in preprocessed", sp.exists())
        if sp.exists():
            splits[d] = sp.read_bytes()
            S = json.loads(sp.read_text())
            check(f"{d}: exactly 3 folds", len(S) == 3, str(len(S)))
            vals = [set(f["val"]) for f in S]
            check(f"{d}: per fold train ∩ val = ∅", all(not (set(f["train"]) & set(f["val"])) for f in S))
            check(f"{d}: val folds partition the train pool", sum(len(v) for v in vals) == len(set().union(*vals)) == len(train[d]) and set().union(*vals) == train[d])
            check(f"{d}: every fold's train∪val == train pool", all(set(f["train"]) | set(f["val"]) == train[d] for f in S))
            check(f"{d}: no test id in any fold", not any((set(f["train"]) | set(f["val"])) & test[d] for f in S))
        pp = prep / d / "gt_segmentations"
        if pp.exists():
            pids = {f.name[:-7] for f in pp.glob("*.nii.gz")}
            check(f"{d}: preprocessed gt_segmentations ⊆ train pool (no test case preprocessed)", pids <= train[d] and not (pids & test[d]), str(sorted(pids - train[d])[:5]))
    if len(dsets) > 1:
        d0 = dsets[0]
        check("same TRAIN id set in every contrast dataset", all(train[d] == train[d0] for d in dsets))
        check("same TEST id set in every contrast dataset", all(test[d] == test[d0] for d in dsets))
        check("splits_final.json byte-identical across contrast datasets", all(splits.get(d) == splits.get(d0) for d in dsets))
    tc = Path(a.splits_dir) / "test_cases.json"
    if tc.exists():
        check("test_cases.json == test set", set(json.loads(tc.read_text())["test"]) == test[dsets[0]])
    if a.participants:
        rows = [ln.split("\t")[0] for ln in Path(a.participants).read_text().splitlines()[1:] if ln.strip()]
        check("participants.tsv subject ids unique", len(rows) == len(set(rows)))
    print(f"\n{len(dsets)} datasets | train {len(train[dsets[0]])} | test {len(test[dsets[0]])} | {'ALL CHECKS PASSED' if not bad else 'FAILED: ' + '; '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
