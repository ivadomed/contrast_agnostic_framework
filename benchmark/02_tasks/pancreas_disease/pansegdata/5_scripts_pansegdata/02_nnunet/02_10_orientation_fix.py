#!/usr/bin/env python3
"""Decision table for the PanSegData source-orientation defect, plus checks that the correction is right.
Defect (found 2026-10-04): in the BIDS leaf (a faithful copy) the voxel arrays of most MCF (Mayo) venous-phase T1 scans are stored rotated 180 degrees in-plane
relative to what their header says (header LPS, array RAS-like; the 66 mismatching label headers RAI/RAS are the same rotation), so after the header-based reorientation to LPS
they are posterior-up and left-right mirrored relative to every other scan. The T1<->T2 mask agreement of MCF same-shape pairs is 16% before and 91% after correcting it.
Rule (from 02_09's CNN, whose A-P bit is reliable but whose L-R bit is ~92%): ONLY for center MCF, p_apflip >= 0.8 -> rotate the scan 180 deg in-plane (flip L-R and A-P together);
p_apflip <= 0.2 -> leave; in between = uncertain -> the SUBJECT (both contrasts) is excluded. Other centers are left untouched (no systematic defect found; a CNN 'flip' there is noise).
Writes 0_raw_pansegdata/orientation_fix.tsv (case, site, contrast, status, rot180, p_apflip) and 9_tests_pansegdata/orientation_fix_qc.png (corrected MCF pairs, axial + coronal).
Also tests S-I (z): same-shape pairs, mask-centroid z fraction T1 vs T2 vs the z-flipped alternative (the in-plane CNN cannot see an upside-down volume)."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
TESTS = DS / "9_tests_pansegdata"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
site = part["site"]; cases = part["train_pool"] + part["test"]
cnn = {(r["case"], r["contrast"]): r for r in csv.DictReader(open(TESTS / "orientation_classifier.tsv"), delimiter="\t")}

rows = []
for (c, it), r in sorted(cnn.items()):
    p = float(r["p_apflip"])
    if site[c] != "MCF": st, rot = "canonical", False
    elif p >= 0.8: st, rot = "rotate180", True
    elif p <= 0.2: st, rot = "canonical", False
    else: st, rot = "uncertain", False
    rows.append(dict(case=c, site=site[c], contrast=it, status=st, rot180=rot, p_apflip=p, cnn_pred=r["pred"]))
unc = sorted({r["case"] for r in rows if r["status"] == "uncertain"})
for r in rows:
    if r["case"] in unc and r["status"] != "uncertain": r["status"] = "partner_uncertain"
with open(DS / "0_raw_pansegdata" / "orientation_fix.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
for it in ("t1wce", "t2w"):
    print(it, "MCF:", dict(collections.Counter(r["status"] for r in rows if r["site"] == "MCF" and r["contrast"] == it)))
print("MCF subjects excluded for uncertain orientation:", len(unc), unc[:12])
rot = {(r["case"], r["contrast"]): r["rot180"] for r in rows}


def lps(arr, affine):
    return apply_orientation(arr, ornt_transform(io_orientation(affine), axcodes2ornt(("L", "P", "S"))))


def load(c, it):
    sub = "sub-" + c.split("pansegdata_")[1]
    stem, suf = (f"{sub}_acq-venous", "T1w") if it == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    x = lps(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    m = lps(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, img.affine)
    if rot[(c, it)]: x, m = x[::-1, ::-1, :], m[::-1, ::-1, :]
    return x, m, np.array(img.header.get_zooms()[:3])[[int(a) for a in io_orientation(img.affine)[:, 0]]]


# ---- S-I (z) consistency of same-shape pairs, and in-plane agreement after the fix
print("same-shape pairs after the fix: center n | in-plane d<0.06 | z-fraction |dz|<0.08 as stored vs z-flipped alternative")
zres = {}
for s in ("NYU", "AHN", "MCF"):
    ids = [c for c in cases if site[c] == s and c not in unc]
    n = a = zo = zf = 0
    for c in ids:
        x1, m1, _ = load(c, "t1wce"); x2, m2, _ = load(c, "t2w")
        if x1.shape != x2.shape: continue
        c1, c2 = np.argwhere(m1).mean(0) / np.array(x1.shape), np.argwhere(m2).mean(0) / np.array(x2.shape)
        n += 1; a += np.linalg.norm((c1 - c2)[:2]) < 0.06
        zo += abs(c1[2] - c2[2]) < 0.08; zf += abs(c1[2] - (1 - c2[2])) < 0.08
    print(f"  {s}: n={n} | {a / n:.2f} | z as stored {zo / n:.2f} vs z-flipped {zf / n:.2f}")

# ---- visual QC of corrected MCF pairs
rng = np.random.default_rng(1)
mcf = [c for c in cases if site[c] == "MCF" and c not in unc]
rotated = [c for c in mcf if rot[(c, "t1wce")]]; plain = [c for c in mcf if not rot[(c, "t1wce")]]
pick = list(rng.choice(rotated, 6, replace=False)) + list(rng.choice(plain, 2, replace=False))
fig, axs = plt.subplots(len(pick), 4, figsize=(8, 2.3 * len(pick)))
for i, c in enumerate(pick):
    for j, it in enumerate(("t1wce", "t2w")):
        x, m, z = load(c, it); cen = np.round(np.argwhere(m).mean(0)).astype(int); lo, hi = np.percentile(x, [1, 99.5])
        a = axs[i][2 * j]; a.imshow(x[:, :, cen[2]].T, cmap="gray", vmin=lo, vmax=hi, origin="upper", aspect=z[1] / z[0]); a.contour(m[:, :, cen[2]].T.astype(float), [0.5], colors="r", linewidths=0.5, origin="upper")
        a.set_title(f"{c[11:]} {it} axial {'ROT180' if rot[(c, it)] else 'as stored'}", fontsize=6); a.axis("off")
        b = axs[i][2 * j + 1]; b.imshow(x[:, cen[1], :].T, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=z[2] / z[0]); b.contour(m[:, cen[1], :].T.astype(float), [0.5], colors="r", linewidths=0.5, origin="lower")
        b.set_title("coronal", fontsize=6); b.axis("off")
fig.suptitle("MCF after the 180-degree fix: axial anterior UP, liver image-left; coronal superior up. T1 and T2 of one subject side by side", fontsize=7)
fig.tight_layout(); fig.savefig(TESTS / "orientation_fix_qc.png", dpi=90)
print("wrote QC png; rotated subjects shown:", [c[11:] for c in pick])
