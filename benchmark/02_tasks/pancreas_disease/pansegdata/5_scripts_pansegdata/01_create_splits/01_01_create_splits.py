#!/usr/bin/env python3
"""PanSegData patient-level partition + 3-fold CV (no leakage). One BIDS subject = one case id shared by BOTH contrasts.

Usable = subjects that have BOTH a venous-phase T1W and a T2W scan (362 of 405) AND are not from the excluded center MCF (see EXCLUDE_SITES below): 212 subjects
(NYU 161, NWU 19, AHN 17, MCA 15). The 43 single-contrast subjects are EXCLUDED from the benchmark
(Paul-autonomy decision 2026-10-04; alternative: keep them train-only, which needs a looser shared verify_heldout): the T1<->T2 pairing was inferred
from original case names (pairing evidence in 9_tests_pansegdata/), so an unlinked twin of a held-out patient could sit among the single-contrast
scans, and the shared verifier requires the identical train id set in both contrast datasets anyway.
Test = N_TEST subjects, allocated to the 5 centers proportionally (largest remainder) and chosen within each center at evenly spaced ranks of mean
pancreas volume, never seen in any fold. The rest is the train pool, split into 3 folds (project policy: folds 0 1 2 only), blocks of 3 consecutive
subjects in (center, volume) order dealt across the folds so every fold has a similar center mix and volume distribution.
Case ids are 'pansegdata_<site><NNNN>' (the BIDS sub- id without the dash), NO contrast suffix.
Writes 4_splits_pansegdata/: partition.json, splits_final.json, test_cases.json. Self-checks leakage before writing."""
from __future__ import annotations
import csv, json, collections
from pathlib import Path
import numpy as np

SEED, N_TEST, N_FOLDS = 1337, 42, 3
# Fixed BEFORE any training. The whole Mayo-Florida center (MCF, 150 paired subjects) is excluded: its venous-phase T1 arrays are stored rotated 180 degrees in-plane
# (and, for many, flipped along S-I) relative to their headers (see 9_tests_pansegdata/ and 0_raw_pansegdata/orientation_fix.tsv), the 66 mismatching label headers
# (RAI/RAS vs LPS) show the same defect, and in the MCF subjects inspected at full size the pancreas mask lies over the vertebral body / paraspinal muscles, i.e. mask and
# image are not reliably registered even after a 180-degree correction. A per-scan rescue could not be verified, so the center is dropped (alternative: rescue/curate).
EXCLUDE_SITES = {"MCF"}
DS = Path(__file__).resolve().parents[2]
AUDIT = DS / "0_raw_pansegdata" / "pansegdata_raw_audit.tsv"
PART = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata" / "participants.tsv"
SPLITS = DS / "4_splits_pansegdata"


def case_id(sub):
    return "pansegdata_" + sub.replace("sub-", "")


def main():
    parts = list(csv.DictReader(open(PART), delimiter="\t"))
    vol = collections.defaultdict(dict)
    for r in csv.DictReader(open(AUDIT), delimiter="\t"):
        vol[r["sub"]][r["contrast"]] = float(r["mask_ml"])
    site, usable, excluded = {}, [], []
    for p in parts:
        s = p["participant_id"]
        if p["pairing"] == "paired" and len(vol[s]) == 2 and min(vol[s].values()) > 0 and p["site"] not in EXCLUDE_SITES:
            usable.append(case_id(s)); site[case_id(s)] = p["site"]
        else:
            excluded.append(case_id(s))
    mv = {c: float(np.mean(list(vol["sub-" + c.split("pansegdata_")[1]].values()))) for c in usable}
    # --- test allocation per site (largest remainder), evenly spaced volume ranks within site
    by_site = collections.defaultdict(list)
    for c in usable: by_site[site[c]].append(c)
    raw = {s: N_TEST * len(v) / len(usable) for s, v in by_site.items()}
    alloc = {s: int(np.floor(x)) for s, x in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - alloc[s], reverse=True)[: N_TEST - sum(alloc.values())]: alloc[s] += 1
    test = []
    for s, cs in sorted(by_site.items()):
        cs = sorted(cs, key=lambda c: (mv[c], c)); k = alloc[s]
        idx = sorted({int(round(i)) for i in np.linspace(0, len(cs) - 1, k)}) if k else []
        j = 0
        while len(idx) < k:
            if j not in idx: idx.append(j)
            j += 1
        test += [cs[i] for i in sorted(idx)]
    test = sorted(test); tset = set(test)
    pool = sorted(c for c in usable if c not in tset)
    rng = np.random.default_rng(SEED)
    order = sorted(pool, key=lambda c: (site[c], mv[c], c))
    folds = [[] for _ in range(N_FOLDS)]
    for b in range(0, len(order), N_FOLDS):
        blk = order[b:b + N_FOLDS]; rng.shuffle(blk)
        for k, c in enumerate(blk): folds[k].append(c)
    splits = [{"train": sorted(c for c in pool if c not in set(f)), "val": sorted(f)} for f in folds]
    # ---- leakage self-check
    assert len(test) == N_TEST and len(set(test)) == N_TEST and not tset & set(pool)
    for s in splits: assert not set(s["train"]) & set(s["val"]) and not (set(s["train"]) | set(s["val"])) & tset
    assert sorted(sum((s["val"] for s in splits), [])) == pool
    assert set(usable) == set(pool) | tset and not set(usable) & set(excluded)
    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps({
        "train_pool": pool, "test": test, "excluded": sorted(excluded),
        "excluded_reason": "(a) no T1<->T2 counterpart found by original case name within the site (23 T1-only + 20 T2-only subjects): a benchmark case needs both contrasts; "
                           "(b) every subject of center MCF (Mayo Florida): image header orientation unreliable (180-degree in-plane / S-I) and masks not reliably registered to the images",
        "site": site, "mean_mask_ml": {c: round(v, 2) for c, v in mv.items()}, "seed": SEED, "test_alloc_per_site": alloc}, indent=2))
    (SPLITS / "splits_final.json").write_text(json.dumps(splits, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({"test": test}, indent=2))
    print("usable", len(usable), "excluded", len(excluded), "pool", len(pool), "test", len(test), "alloc", alloc)
    for k, s in enumerate(splits):
        sc = collections.Counter(site[c] for c in s["val"])
        print(f"fold {k}: {len(s['train'])} train / {len(s['val'])} val; val median ml {np.median([mv[c] for c in s['val']]):.1f}; val sites {dict(sorted(sc.items()))}")
    print(f"test median ml {np.median([mv[c] for c in test]):.1f} vs pool {np.median([mv[c] for c in pool]):.1f}; test sites {dict(sorted(collections.Counter(site[c] for c in test).items()))}")


if __name__ == "__main__":
    main()
