#!/usr/bin/env python3
"""ISLES'22 patient-level partition + 3-fold CV (no leakage). One subject = one case (1 session each).

Usable = non-empty mask AND FLAIR FOV containing the lesion (246 of 250). Test = N_TEST cases chosen STRATIFIED by lesion volume
(evenly spaced ranks), never seen in any fold; the rest is the train pool, split into 3 folds
(project fold policy: folds 0 1 2 only). Case ids are 'isles2022_<NNNN>' with NO contrast suffix, so
every contrast of a patient shares one id (the patient-level significance merge pairs them).
Scanner/site is not in the release (JSON present for only some cases), so site cannot be stratified on.

Writes 4_splits_isles2022/: partition.json, splits_final.json, test_cases.json."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

SEED, N_TEST, N_FOLDS = 1337, 47, 3
# Fixed BEFORE any training. Found by 02_01_convert.py's QC: FLAIR field of view contains only 53% of this case's
# lesion (every other case >=97%), so the FLAIR contrast cannot represent its label; all contrasts must share one
# case set, so it is dropped from the whole task.
EXCLUDE_FLAIR_FOV = {"isles2022_0007"}
DS = Path(__file__).resolve().parents[2]
AUDIT = DS / "0_raw_isles2022" / "isles2022_raw_audit.tsv"
SPLITS = DS / "4_splits_isles2022"

def main():
    rows = list(csv.DictReader(open(AUDIT), delimiter="\t"))
    vol, excluded = {}, []
    for r in rows:
        cid = "isles2022_" + r["case"].replace("sub-strokecase", "")
        if int(r["n_vox"]) == 0 or cid in EXCLUDE_FLAIR_FOV:
            excluded.append(cid); continue
        vol[cid] = float(r["lesion_ml"])
    cases = sorted(vol)
    by = sorted(cases, key=lambda c: (vol[c], c))
    idx = np.linspace(0, len(by) - 1, N_TEST).round().astype(int)
    test = sorted({by[i] for i in idx})
    i = 0
    while len(test) < N_TEST:
        test = sorted(set(test) | {by[i]}); i += 1
    pool = sorted(c for c in cases if c not in set(test))
    rng = np.random.default_rng(SEED)
    sh = list(pool); rng.shuffle(sh)
    # volume-balanced folds: sort by volume, deal round-robin within the seeded shuffle's rank blocks
    sh = sorted(sh, key=lambda c: (vol[c], c))
    folds = [[] for _ in range(N_FOLDS)]
    for b in range(0, len(sh), N_FOLDS):
        blk = sh[b:b + N_FOLDS]; rng.shuffle(blk)
        for k, c in enumerate(blk): folds[k].append(c)
    splits = [{"train": sorted(c for c in pool if c not in set(f)), "val": sorted(f)} for f in folds]
    for s in splits: assert not set(s["train"]) & set(s["val"]) and not (set(s["train"]) | set(s["val"])) & set(test)
    assert sorted(sum((s["val"] for s in splits), [])) == pool
    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps({"train_pool": pool, "test": test, "excluded": excluded, "excluded_reason": "empty mask (0150, 0151, 0170); FLAIR FOV cuts lesion (0007)",
        "lesion_ml": vol, "seed": SEED}, indent=2))
    (SPLITS / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({"test": test}, indent=2))
    print("usable", len(cases), "excluded", excluded, "pool", len(pool), "test", len(test))
    for k, s in enumerate(splits):
        print(f"fold {k}: {len(s['train'])} train / {len(s['val'])} val; val median ml {np.median([vol[c] for c in s['val']]):.2f}")
    print(f"test median ml {np.median([vol[c] for c in test]):.2f} vs pool {np.median([vol[c] for c in pool]):.2f}")

if __name__ == "__main__":
    main()
