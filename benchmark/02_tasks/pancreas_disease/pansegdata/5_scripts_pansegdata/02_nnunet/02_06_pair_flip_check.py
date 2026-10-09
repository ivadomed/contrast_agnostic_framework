#!/usr/bin/env python3
"""Pair flip test: for each subject whose T1WCE and T2W arrays have the SAME shape (all of NYU, AHN, MCF), compare the T1-mask and T2-mask
positions in array space (both reoriented to LPS with their image affine), for the T2 mask as stored and flipped along A-P / L-R / both.
Distance = centroid difference in units of the array extent (so FOV differences between centers do not matter; same-shape pairs share a matrix).
If the stored T2 mask is aligned like the (flip-test-verified) T1 mask, 'orig' should be the smallest; a case where a flipped variant fits much better
is a candidate misaligned mask. Also the polarity-correct T2 contrast test: on T2 the pancreas is darker than the surrounding fat, so the stored mask
should have a LOWER ring contrast than its flips (the T1 test expects higher).
Writes 9_tests_pansegdata/pair_flip_test.tsv + prints the per-center summary and the flagged cases."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy import ndimage as ndi

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
OUT = DS / "9_tests_pansegdata" / "pair_flip_test.tsv"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())


def lps(arr, affine):
    return apply_orientation(arr, ornt_transform(io_orientation(affine), axcodes2ornt(("L", "P", "S"))))


def load(sub, item):
    stem, suf = (f"{sub}_acq-venous", "T1w") if item == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    x = lps(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    m = lps(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, img.affine)
    return x, m


def cnr(x, m):
    idx = np.argwhere(m); lo, hi = idx.min(0), idx.max(0)
    sl = tuple(slice(max(0, lo[i] - 8), hi[i] + 9) for i in range(3))
    xs, ms = x[sl], m[sl]
    ring = ndi.binary_dilation(ms, iterations=4) & ~ndi.binary_dilation(ms, iterations=1)
    return float((xs[ms].mean() - xs[ring].mean()) / (xs[ring].std() + 1e-6)) if ring.sum() > 10 else float("nan")


def cen(m):
    return np.argwhere(m).mean(0) / np.array(m.shape)


rows = []
for cid in part["train_pool"] + part["test"]:
    sub = "sub-" + cid.split("pansegdata_")[1]
    x1, m1 = load(sub, "t1wce"); x2, m2 = load(sub, "t2w")
    r = dict(case=cid, site=part["site"][cid], same_shape=bool(x1.shape == x2.shape))
    flips = {"orig": m2, "flipAP": m2[:, ::-1, :], "flipLR": m2[::-1, :, :], "flipBoth": m2[::-1, ::-1, :]}
    if r["same_shape"]:
        d = {k: float(np.linalg.norm((cen(v) - cen(m1))[:2])) for k, v in flips.items()}   # in-plane (L-R, A-P) distance, fraction of extent
        r.update({f"d_{k}": round(v, 3) for k, v in d.items()}); r["best"] = min(d, key=d.get)
    c2 = {k: cnr(x2, v) for k, v in flips.items()}
    r["t2_cnr_orig"] = round(c2["orig"], 3); r["t2_stored_lowest_cnr"] = bool(c2["orig"] <= min(c2["flipAP"], c2["flipLR"], c2["flipBoth"]))
    c1 = {"orig": cnr(x1, m1), "flipAP": cnr(x1, m1[:, ::-1, :]), "flipLR": cnr(x1, m1[::-1, :, :]), "flipBoth": cnr(x1, m1[::-1, ::-1, :])}
    r["t1_stored_highest_cnr"] = bool(c1["orig"] >= max(c1["flipAP"], c1["flipLR"], c1["flipBoth"]))
    rows.append(r)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, keys, delimiter="\t", lineterminator="\n", restval="n/a"); w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows: g[r["site"]].append(r)
print("center n same_shape | pair: orig best | flipAP best | flipLR best | flipBoth best | median d_orig | T1 stored highest cnr | T2 stored lowest cnr")
for s in sorted(g):
    v = g[s]; ss = [r for r in v if r["same_shape"]]
    pb = collections.Counter(r["best"] for r in ss)
    med = np.median([r["d_orig"] for r in ss]) if ss else float("nan")
    print(f"{s:4s} {len(v):4d} {len(ss):4d} | {pb.get('orig',0):4d} {pb.get('flipAP',0):4d} {pb.get('flipLR',0):4d} {pb.get('flipBoth',0):4d} | {med:.3f} | "
          f"{np.mean([r['t1_stored_highest_cnr'] for r in v]):.2f} | {np.mean([r['t2_stored_lowest_cnr'] for r in v]):.2f}")
flag = [r["case"][11:] for r in rows if r["same_shape"] and r["best"] != "orig" and r["d_orig"] - r[f"d_{r['best']}"] > 0.10]
print("cases where a flipped T2 mask fits the T1 mask better by >0.10:", len(flag), flag[:40])
