#!/usr/bin/env python3
"""Large, labelled renders of a few subjects to judge by eye where each mask sits relative to the anatomy, uncorrected (header LPS) vs corrected (02_10 table).
One PNG per subject in 9_tests_pansegdata/look_<case>.png. Row 1 = T1WCE: axial as header-LPS, axial corrected, coronal corrected. Row 2 = T2W: axial, coronal, sagittal
(corrected). Arrays are LPS; axial drawn with anterior UP and patient-left on the image RIGHT (labels A/P/R/L drawn on the panel).
Usage: python 02_11_look_closely.py <case> [<case> ...]   (default: a fixed list)"""
import csv, sys
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
TESTS = DS / "9_tests_pansegdata"
fix = {(r["case"], r["contrast"]): r["rot180"] == "True" for r in csv.DictReader(open(DS / "0_raw_pansegdata" / "orientation_fix.tsv"), delimiter="\t")}
cases = sys.argv[1:] or ["pansegdata_mcf0074", "pansegdata_mcf0083", "pansegdata_mcf0141", "pansegdata_nyu0100"]


def load(c, it, corrected):
    sub = "sub-" + c.split("pansegdata_")[1]
    stem, suf = (f"{sub}_acq-venous", "T1w") if it == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    t = ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "P", "S")))
    x = apply_orientation(np.asanyarray(img.dataobj).astype(np.float32), t)
    m = apply_orientation(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, t)
    if corrected and fix[(c, it)]: x, m = x[::-1, ::-1, :], m[::-1, ::-1, :]
    return x, m, np.array(img.header.get_zooms()[:3])[[int(a) for a in io_orientation(img.affine)[:, 0]]], "".join(nib.aff2axcodes(img.affine))


def draw(ax, x, m, z, view, title):
    cen = np.round(np.argwhere(m).mean(0)).astype(int); lo, hi = np.percentile(x, [1, 99.7])
    if view == "axial": sl, ml, org, asp = x[:, :, cen[2]].T, m[:, :, cen[2]].T, "upper", z[1] / z[0]
    elif view == "coronal": sl, ml, org, asp = x[:, cen[1], :].T, m[:, cen[1], :].T, "lower", z[2] / z[0]
    else: sl, ml, org, asp = x[cen[0], :, :].T, m[cen[0], :, :].T, "lower", z[2] / z[1]
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin=org, aspect=asp); ax.contour(ml.astype(float), [0.5], colors="r", linewidths=0.8, origin=org)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    lab = {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"), "sagittal": ("A", "P", "S", "I")}[view]
    ax.text(0.02, 0.5, lab[0], transform=ax.transAxes, color="yellow", fontsize=9); ax.text(0.95, 0.5, lab[1], transform=ax.transAxes, color="yellow", fontsize=9)
    ax.text(0.5, 0.93, lab[2], transform=ax.transAxes, color="yellow", fontsize=9); ax.text(0.5, 0.02, lab[3], transform=ax.transAxes, color="yellow", fontsize=9)


for c in cases:
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    x, m, z, ax_ = load(c, "t1wce", False); draw(axs[0][0], x, m, z, "axial", f"{c[11:]} T1 header-LPS (stored axcodes {ax_})")
    x, m, z, _ = load(c, "t1wce", True); draw(axs[0][1], x, m, z, "axial", f"T1 corrected (rot180={fix[(c, 't1wce')]})"); draw(axs[0][2], x, m, z, "coronal", "T1 corrected coronal")
    x, m, z, ax2 = load(c, "t2w", True); draw(axs[1][0], x, m, z, "axial", f"T2 corrected (rot180={fix[(c, 't2w')]}, axcodes {ax2})"); draw(axs[1][1], x, m, z, "coronal", "T2 coronal"); draw(axs[1][2], x, m, z, "sagittal", "T2 sagittal")
    fig.tight_layout(); fig.savefig(TESTS / f"look_{c[11:]}.png", dpi=75); plt.close(fig); print("wrote", c)
