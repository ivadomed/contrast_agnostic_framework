#!/usr/bin/env python3
"""Per-scan audit of the PanSegData BIDS leaf (faithful copy of the source) -> 0_raw_pansegdata/pansegdata_raw_audit.tsv (one row per scan).
Columns: sub, contrast (t1wce|t2w), site, pairing, shape, spacing_mm, raw_axcodes, label_axcodes_ok, mask_ml, n_components, touches_border, cnr.
cnr = (mean intensity inside the mask - mean in a 3-voxel ring outside it) / std of the ring; physics check: positive on venous-phase T1
(pancreas enhances), mostly negative on T2 (pancreas darker than the surrounding fat). Rows feed 01_01_create_splits.py (organ volume, pairing).
Written with LF line endings (a CRLF TSV silently breaks column lookups in awk)."""
import csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
OUT = DS / "0_raw_pansegdata" / "pansegdata_raw_audit.tsv"

rows = []
for r in csv.DictReader(open(BIDS / "participants.tsv"), delimiter="\t"):
    sub = r["participant_id"]
    for contrast, stem, suf, have in (("t1wce", f"{sub}_acq-venous", "T1w", r["source_t1_id"] != "n/a"), ("t2w", sub, "T2w", r["source_t2_id"] != "n/a")):
        if not have:
            continue
        img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
        lab = nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz"))
        x = np.asanyarray(img.dataobj, dtype=np.float32); m = np.asanyarray(lab.dataobj) > 0
        z = img.header.get_zooms()[:3]
        idx = np.argwhere(m)
        lo, hi = idx.min(0), idx.max(0)
        sl = tuple(slice(max(0, lo[i] - 8), hi[i] + 9) for i in range(3))
        xs, ms = x[sl], m[sl]
        ring = ndi.binary_dilation(ms, iterations=4) & ~ndi.binary_dilation(ms, iterations=1)
        cnr = (xs[ms].mean() - xs[ring].mean()) / (xs[ring].std() + 1e-6) if ring.sum() > 10 else float("nan")
        rows.append(dict(sub=sub, contrast=contrast, site=r["site"], pairing=r["pairing"], shape="x".join(map(str, img.shape)),
                         spacing_mm="x".join(f"{v:.3f}" for v in z), raw_axcodes="".join(nib.aff2axcodes(img.affine)),
                         label_on_image_grid=bool(lab.shape == img.shape and np.allclose(lab.affine, img.affine, atol=1e-3)),
                         mask_ml=round(float(m.sum() * np.prod(z) / 1000), 2), n_components=int(ndi.label(m)[1]),
                         touches_border=bool(lo.min() == 0 or any(hi[i] == img.shape[i] - 1 for i in range(3))), cnr=round(float(cnr), 3)))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
n1 = [r for r in rows if r["contrast"] == "t1wce"]; n2 = [r for r in rows if r["contrast"] == "t2w"]
print(f"scans {len(rows)} (t1wce {len(n1)}, t2w {len(n2)}); empty masks {sum(r['mask_ml'] == 0 for r in rows)}; "
      f"mean cnr t1wce {np.nanmean([r['cnr'] for r in n1]):.2f} t2w {np.nanmean([r['cnr'] for r in n2]):.2f}; "
      f"label off image grid {sum(not r['label_on_image_grid'] for r in rows)}; touching border {sum(r['touches_border'] for r in rows)}")
