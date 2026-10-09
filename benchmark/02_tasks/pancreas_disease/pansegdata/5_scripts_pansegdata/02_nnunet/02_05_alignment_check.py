#!/usr/bin/env python3
"""Is every pancreas mask spatially ALIGNED with its image? (aff2axcodes and a few eyeballed slices cannot prove this for 724 scans.)
All arrays reoriented in memory to LPS using the IMAGE affine (the same op the converter applies). Per scan:
  pfrac/lfrac = mask centroid position as a fraction of the body extent along A-P (0 anterior .. 1 posterior) and L-R;
  frac_post   = fraction of mask voxels in the posterior 25% of the body (a pancreas cannot be there: spine/back muscle);
  cnr0 / cnr_flip = intensity contrast (mask mean - ring mean)/ring std for the mask as stored and for the mask flipped along A-P, L-R, both;
                    on venous-phase T1 (pancreas enhances) the stored mask should beat every flip;
  pair agreement  = for subjects with both contrasts, |dp|,|dl| between the T1 and T2 mask centroids in body-normalised coordinates.
Also asserts the converted nnU-Net files (2_nnUNet_pansegdata/raw) equal the BIDS arrays reoriented in memory (conversion consistency).
Writes 9_tests_pansegdata/alignment_check.tsv (LF) + prints a per-center x contrast summary."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy import ndimage as ndi

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
RAW = DS / "2_nnUNet_pansegdata" / "raw"
OUT = DS / "9_tests_pansegdata" / "alignment_check.tsv"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
cases = part["train_pool"] + part["test"]
STEM = {"t1wce": ("{sub}_acq-venous", "T1w", "Dataset150_PanSegData_T1WCE"), "t2w": ("{sub}", "T2w", "Dataset151_PanSegData_T2W")}


def lps(arr, affine):
    t = ornt_transform(io_orientation(affine), axcodes2ornt(("L", "P", "S")))
    return apply_orientation(arr, t)


def cnr(x, m):
    idx = np.argwhere(m); lo, hi = idx.min(0), idx.max(0)
    sl = tuple(slice(max(0, lo[i] - 8), hi[i] + 9) for i in range(3))
    xs, ms = x[sl], m[sl]
    ring = ndi.binary_dilation(ms, iterations=4) & ~ndi.binary_dilation(ms, iterations=1)
    return float((xs[ms].mean() - xs[ring].mean()) / (xs[ring].std() + 1e-6)) if ring.sum() > 10 else float("nan")


rows, cen = [], {}
for cid in cases:
    sub = "sub-" + cid.split("pansegdata_")[1]; site = part["site"][cid]
    for item, (stem, suf, ds) in STEM.items():
        stem = stem.format(sub=sub)
        img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
        x = lps(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
        m = lps(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, img.affine)
        body = x > 0.10 * np.percentile(x, 99)
        pa = np.where(body.any((0, 2)))[0]; la = np.where(body.any((1, 2)))[0]
        idx = np.argwhere(m)
        pf = (idx[:, 1].mean() - pa.min()) / max(pa.max() - pa.min(), 1); lf = (idx[:, 0].mean() - la.min()) / max(la.max() - la.min(), 1)
        post = float((idx[:, 1] > pa.min() + 0.75 * (pa.max() - pa.min())).mean())
        c0 = cnr(x, m)
        flips = {"flipAP": cnr(x, m[:, ::-1, :]), "flipLR": cnr(x, m[::-1, :, :]), "flipBoth": cnr(x, m[::-1, ::-1, :])}
        best = max(flips.values())
        # converted file == BIDS array reoriented in memory?
        sp = "imagesTr" if cid in set(part["train_pool"]) and True else None
        cdir = RAW / ds / ("imagesTr" if cid in set(part["train_pool"]) else f"imagesTs_{item}")
        conv = np.asanyarray(nib.load(str(cdir / f"{cid}_0000.nii.gz")).dataobj)
        same = bool(conv.shape == x.shape and np.allclose(conv, x))
        cen[(cid, item)] = (pf, lf)
        rows.append(dict(case=cid, site=site, contrast=item, pfrac=round(pf, 3), lfrac=round(lf, 3), frac_post=round(post, 3), cnr0=round(c0, 3),
                         cnr_best_flip=round(best, 3), stored_beats_flips=bool(c0 >= best), converted_equals_bids=same))
# pair agreement
for r in rows:
    a, b = cen[(r["case"], "t1wce")], cen[(r["case"], "t2w")]
    r["pair_dp"], r["pair_dl"] = round(abs(a[0] - b[0]), 3), round(abs(a[1] - b[1]), 3)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows: g[(r["site"], r["contrast"])].append(r)
print("center contrast  n | median pfrac | frac_post>0.5 | stored mask beats all flips | pair |dp|>0.15 | conv==bids")
for k in sorted(g):
    v = g[k]
    print(f"{k[0]:4s} {k[1]:5s} {len(v):4d} | {np.median([r['pfrac'] for r in v]):.2f} | {np.mean([r['frac_post'] > 0.5 for r in v]):.2f} | "
          f"{np.mean([r['stored_beats_flips'] for r in v]):.2f} | {np.mean([r['pair_dp'] > 0.15 for r in v]):.2f} | {np.mean([r['converted_equals_bids'] for r in v]):.2f}")
bad = [r for r in rows if r["frac_post"] > 0.5]
print("scans with >50% of the mask in the posterior quarter of the body:", len(bad), [(r["case"][11:], r["contrast"]) for r in bad][:25])
print("converted != bids:", sum(not r["converted_equals_bids"] for r in rows))
