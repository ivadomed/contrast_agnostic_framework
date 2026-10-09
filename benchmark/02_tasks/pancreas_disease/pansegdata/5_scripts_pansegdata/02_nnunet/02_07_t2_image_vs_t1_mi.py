#!/usr/bin/env python3
"""Which array is rotated? For same-shape T1WCE/T2W pairs (both reoriented to LPS with their own image affine) compute the mutual information between the
T1 IMAGE and the T2 IMAGE as stored and with the T2 image flipped along A-P / L-R / both (MI is sequence-agnostic, so it can register T1 to T2).
Cross-tab with the mask-based best variant from 02_06_pair_flip_check (pair_flip_test.tsv):
  image best == mask best != orig  -> T2 image AND mask are rotated together relative to T1 (a header/orientation problem of the T2 acquisition)
  image best == orig, mask best != orig -> only the T2 MASK is misplaced relative to its own image (a labelling problem)
Writes 9_tests_pansegdata/t2_vs_t1_mi.tsv and prints the cross-tab per center."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
TESTS = DS / "9_tests_pansegdata"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
pf = {r["case"]: r for r in csv.DictReader(open(TESTS / "pair_flip_test.tsv"), delimiter="\t")}


def lps_img(sub, item):
    stem, suf = (f"{sub}_acq-venous", "T1w") if item == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    return apply_orientation(np.asanyarray(img.dataobj).astype(np.float32), ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "P", "S"))))


def mi(a, b, bins=24):
    ok = (a > np.percentile(a, 20)) | (b > np.percentile(b, 20))
    a, b = a[ok][::3], b[ok][::3]
    a = np.clip(a, *np.percentile(a, [1, 99])); b = np.clip(b, *np.percentile(b, [1, 99]))
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / h.sum(); px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


rows = []
for cid in part["train_pool"] + part["test"]:
    r = pf[cid]
    if r["same_shape"] != "True":
        continue
    sub = "sub-" + cid.split("pansegdata_")[1]
    x1, x2 = lps_img(sub, "t1wce"), lps_img(sub, "t2w")
    v = {"orig": x2, "flipAP": x2[:, ::-1, :], "flipLR": x2[::-1, :, :], "flipBoth": x2[::-1, ::-1, :]}
    m = {k: mi(x1, a) for k, a in v.items()}
    best = max(m, key=m.get); srt = sorted(m.values())
    rows.append(dict(case=cid, site=r["site"], **{f"mi_{k}": round(val, 4) for k, val in m.items()}, img_best=best, img_margin=round(srt[-1] - srt[-2], 4), mask_best=r["best"]))
with open(TESTS / "t2_vs_t1_mi.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
for s in ("NYU", "AHN", "MCF"):
    v = [r for r in rows if r["site"] == s]
    ct = collections.Counter((r["img_best"], r["mask_best"]) for r in v)
    print(f"{s} n={len(v)}  T2-image best variant: {dict(collections.Counter(r['img_best'] for r in v))}")
    print(f"    (T2-image best, T2-mask best) -> count: {dict(sorted(ct.items(), key=lambda kv: -kv[1]))}")
