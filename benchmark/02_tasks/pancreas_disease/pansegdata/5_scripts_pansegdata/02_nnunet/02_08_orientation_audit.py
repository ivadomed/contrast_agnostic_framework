#!/usr/bin/env python3
"""Absolute in-plane orientation audit of every scan (the source headers are NOT reliable for all MCF scans; aff2axcodes cannot see it).
Method: for each scan (BIDS image reoriented to LPS by its header) take the axial slab at the mask-centroid slice, a 320 x 320 mm window centred on the
BODY centre (Otsu body mask, so it does not depend on the mask), resampled to 32 x 32, intensity-normalised. Per contrast a TEMPLATE is the mean patch of
all NYU scans (NYU orientation checked by eye: anterior up, liver image-left). Every scan is compared to its contrast template under 4 in-plane variants
(as stored / flip A-P / flip L-R / both = 180 deg rotation); the best variant is the scan's orientation correction (+ margin to the 2nd best).
Mask-vs-image consistency AFTER that correction: mask centroid offset from the body centre (mm, in the corrected frame) is compared with the NYU median
('normal') and its 180-degree mirror -> says whether the mask follows its image or is rotated relative to it.
Writes 9_tests_pansegdata/orientation_audit.tsv and prints variant counts per center x contrast and the mask-consistency table."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy import ndimage as ndi

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
TESTS = DS / "9_tests_pansegdata"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
N, WIN = 32, 320.0
VAR = {"orig": (False, False), "flipAP": (True, False), "flipLR": (False, True), "flipBoth": (True, True)}


def lps(arr, affine):
    return apply_orientation(arr, ornt_transform(io_orientation(affine), axcodes2ornt(("L", "P", "S"))))


def otsu(v):
    h, e = np.histogram(v, bins=128); c = (e[:-1] + e[1:]) / 2; w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def scan(sub, item):
    stem, suf = (f"{sub}_acq-venous", "T1w") if item == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    x = lps(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    m = lps(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, img.affine)
    z = np.array(img.header.get_zooms()[:3])[[int(a) for a in io_orientation(img.affine)[:, 0]]]
    return x, m, z


def patch(x, m, z):
    zc = int(round(np.argwhere(m)[:, 2].mean()))
    sl = x[:, :, max(0, zc - 1):zc + 2].mean(2)
    lo, hi = np.percentile(sl, [1, 99.5]); sl = np.clip(sl, lo, hi)
    body = ndi.binary_fill_holes(ndi.gaussian_filter(sl, 2) > otsu(ndi.gaussian_filter(sl, 2).ravel()))
    lab, n = ndi.label(body)
    if n > 1: body = lab == (np.argmax(ndi.sum(body, lab, range(1, n + 1))) + 1)
    bx = np.argwhere(body); cen = (bx.min(0) + bx.max(0)) / 2.0                         # body bbox centre, voxel units (axis0 = L, axis1 = P)
    g = (np.arange(N) - (N - 1) / 2) * (WIN / N)
    ii, jj = np.meshgrid(cen[0] + g / z[0], cen[1] + g / z[1], indexing="ij")
    p = ndi.map_coordinates(sl, [ii, jj], order=1, cval=float(sl.min()))
    p = (p - p.mean()) / (p.std() + 1e-6)
    mc = np.argwhere(m).mean(0)
    off = np.array([(mc[0] - cen[0]) * z[0], (mc[1] - cen[1]) * z[1]])                  # mm: [L-R, A-P(+ = posterior)] from the body centre
    return p, off


def variant(p, v):
    ap, lr = VAR[v]
    q = p[::-1, :] if lr else p      # axis0 = L-R
    return q[:, ::-1] if ap else q   # axis1 = A-P


data = {}
for cid in part["train_pool"] + part["test"]:
    sub = "sub-" + cid.split("pansegdata_")[1]
    for item in ("t1wce", "t2w"):
        x, m, z = scan(sub, item); data[(cid, item)] = patch(x, m, z)
tmpl = {it: np.mean([data[(c, it)][0] for c in part["train_pool"] + part["test"] if part["site"][c] == "NYU"], 0) for it in ("t1wce", "t2w")}
prot = {it: np.median([data[(c, it)][1] for c in part["train_pool"] + part["test"] if part["site"][c] == "NYU"], 0) for it in ("t1wce", "t2w")}
print("NYU mask offset prototype from body centre (mm, [L-R, A-P(+post)]):", {k: np.round(v, 1).tolist() for k, v in prot.items()})
rows = []
for (cid, it), (p, off) in data.items():
    cc = {v: float(np.corrcoef(variant(p, v).ravel(), tmpl[it].ravel())[0, 1]) for v in VAR}
    best = max(cc, key=cc.get); srt = sorted(cc.values())
    ap, lr = VAR[best]
    off_c = off * np.array([-1 if lr else 1, -1 if ap else 1])                           # mask offset in the corrected frame
    d_norm = float(np.linalg.norm(off_c - prot[it])); d_rot = float(np.linalg.norm(off_c + prot[it]))
    rows.append(dict(case=cid, site=part["site"][cid], contrast=it, best_variant=best, margin=round(srt[-1] - srt[-2], 3),
                     corr_orig=round(cc["orig"], 3), mask_consistent=bool(d_norm < d_rot), off_L=round(float(off_c[0]), 1), off_P=round(float(off_c[1]), 1)))
with open(TESTS / "orientation_audit.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows: g[(r["site"], r["contrast"])].append(r)
print("center contrast n | variant counts | median margin | mask consistent with corrected image")
for k in sorted(g):
    v = g[k]
    print(f"{k[0]:4s} {k[1]:5s} {len(v):4d} | {dict(collections.Counter(r['best_variant'] for r in v))} | {np.median([r['margin'] for r in v]):.3f} | {np.mean([r['mask_consistent'] for r in v]):.2f}")
