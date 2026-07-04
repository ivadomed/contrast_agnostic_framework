#!/usr/bin/env python3
"""
Build CT⊕US "chimera" test inputs for the context-injection experiment.

HYPOTHESIS
----------
Chaos models fail on cropped US (see datasets/trusted/README + the center-bias
control) — but maybe because the US field-of-view lacks anatomical CONTEXT, not
because US appearance is impossible. Test: embed the US kidney into a full-torso CT
(which has rich context and is easy to segment) and see if the model can now find
the US kidney.

CONSTRUCTION (per patient that has a CT + ≥1 US side)
-----------------------------------------------------
Base volume = the patient's native CT (full torso). Into it we paste the US
kidney(s), using ONLY two priors (never the US GT for placement — that would be
circular, since segmenting the US kidney is the goal):
  1. CT kidney GT  → WHERE each kidney is (segmenting kidneys in CT is easy). We
     split the binary CT kidney mask into right/left connected components and take
     each component's centroid as the paste location.
  2. "Kidney is ~centered in the US volume" → the US-volume CENTER is the kidney
     anchor. We align that center to the CT-kidney centroid.

US/CT are NOT in a shared world frame (TRUSTED is a registration dataset; the
stored US affine is a per-device frame). So we treat US as a voxel array: resample
it to the CT's voxel spacing, then paste a centered block at the CT-kidney centroid.

Intensity: US (uint8 0–254) is linearly rescaled into the CT's robust HU range
([p1,p99] of the CT) so it blends with the volume the model normalises. We overwrite
CT voxels ONLY where the resampled US has signal (>0), so the US fan replaces tissue
without stamping black-border boxes.

Right/left: assigned from the CT components by L-axis position (LPS axis 0: larger =
patient-left). US side comes from the filename (acq-R / acq-L). A patient may get one
or two kidneys pasted; the eval (06_…chimera) measures ONLY the pasted US region(s)
via the ROI mask, so one-sided patients are scored on their single inserted kidney.

OUTPUTS  (new standardized "chimera" test item, parallel to ct/us)
  2_nnUNet_trusted/raw/imagesTs_chimera/<patient>_0000.nii.gz   chimera image (CT grid)
  2_nnUNet_trusted/raw/labelsTs_chimera/<patient>.nii.gz        placed US kidney GT (eval target)
  2_nnUNet_trusted/raw/roiTs_chimera/<patient>.nii.gz           pasted-US region (eval ROI; 1=R,2=L)
  2_nnUNet_trusted/raw/chimera_manifest.csv                     per-patient sides inserted

0_raw is untouched. Reads the clean BIDS tree (native CT, 0.3 mm US, binarized GTs).

    bash 05_20_build_chimera_inputs.sh      # (preferred — heavy, runs on a node)
    python 05_20_build_chimera_inputs.py
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS = DATASET_ROOT / "1_BIDS_trusted" / "trusted-kidney"
DERIV = BIDS / "derivatives" / "manual_masks"
NN = DATASET_ROOT / "2_nnUNet_trusted" / "raw"

WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("BUILD_WORKERS", "4")))
SIDE_ROI = {"R": 1, "L": 2}            # roi mask values per side


def _components_lr(ct_gt: np.ndarray):
    """Split binary CT kidney mask into right/left centroids (voxel coords).
    Returns {'R': centroid, 'L': centroid} for whichever sides are found.
    LPS axis 0 = L: larger index → patient-left."""
    lab, n = ndimage.label(ct_gt > 0)
    if n == 0:
        return {}
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    comps = [i + 1 for i in order if sizes[i] > 50][:2]      # 2 largest real components
    cents = [np.array(ndimage.center_of_mass(ct_gt > 0, lab, c)) for c in comps]
    if len(cents) == 2:
        # larger axis-0 (L) centroid = left kidney
        left, right = (cents[0], cents[1]) if cents[0][0] > cents[1][0] else (cents[1], cents[0])
        return {"R": right, "L": left}
    if len(cents) == 1:
        # one component → split it at its axis-0 median to get both sides
        c = comps[0]
        idx = np.argwhere(lab == c)
        med = np.median(idx[:, 0])
        rcent = idx[idx[:, 0] <= med].mean(0)
        lcent = idx[idx[:, 0] > med].mean(0)
        return {"R": rcent, "L": lcent}
    return {}


def _resample_to_spacing(arr, src_sp, dst_sp, order):
    zoom = np.array(src_sp, float) / np.array(dst_sp, float)
    # prefilter=False: skip ndimage's spline prefilter, which otherwise upcasts the full
    # ~570 M-voxel US to float64 (~4.5 GB). The prefilter is a no-op for order≤1 anyway,
    # so this is exact for our linear (image) / nearest (mask) resampling and keeps RAM low.
    return ndimage.zoom(arr, zoom, order=order, mode="constant", cval=0, prefilter=False)


def _fan_mask(us):
    """Full US fan, with burnt-in annotations removed. US scans have hardware overlays
    (depth rulers, text, sidebar) burnt in at the FOV edges; these are SEPARATE small
    components floating in the black border. The true fan is the single large connected
    speckle region. So: largest connected component of the signal (+ fill its interior
    anechoic gaps) = full fan, minus the disconnected annotation blobs. No brightness
    heuristic, keeps the whole fan."""
    sig = ndimage.binary_closing(us > 0, iterations=1)   # bridge tiny speckle gaps only
    lab, n = ndimage.label(sig)
    if n == 0:
        return np.zeros(us.shape, bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    fan = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(fan)


def _paste(dst, block, center_vox):
    """Paste `block` into `dst` centered at `center_vox` (voxel). Returns the slice
    tuple actually written (clipped to bounds) and the matching block slices."""
    cs, ds, bs = [], [], []
    for ax in range(3):
        c = int(round(center_vox[ax])); b = block.shape[ax]
        lo = c - b // 2; hi = lo + b
        dlo, dhi = max(0, lo), min(dst.shape[ax], hi)
        blo, bhi = dlo - lo, dlo - lo + (dhi - dlo)
        cs.append(slice(dlo, dhi)); bs.append(slice(blo, bhi))
        if dhi <= dlo:
            return None, None
    return tuple(cs), tuple(bs)


def build_patient(args):
    pid, sides = args
    sub = f"sub-{pid}"
    ct_img = nib.load(str(BIDS / sub / "anat" / f"{sub}_CT.nii.gz"))
    ct = np.asarray(ct_img.dataobj).astype(np.float32)
    ct_sp = ct_img.header.get_zooms()[:3]
    ct_gt = np.asarray(nib.load(str(DERIV / sub / "anat" / f"{sub}_CT_dseg.nii.gz")).dataobj)

    cents = _components_lr(ct_gt)
    if not cents:
        return pid, "no CT kidney components", []

    lo, hi = np.percentile(ct, [1, 99])                  # CT robust HU range for rescale
    chim = ct.copy()
    placed_gt = np.zeros(ct.shape, np.uint8)
    roi = np.zeros(ct.shape, np.uint8)
    inserted = []

    for side in sides:
        if side not in cents:
            continue
        us_img = nib.load(str(BIDS / sub / "anat" / f"{sub}_acq-{side}_US.nii.gz"))
        us = np.asarray(us_img.dataobj)                  # keep uint8 (no float upcast → ~4× less RAM)
        us_sp = us_img.header.get_zooms()[:3]
        us_gt = np.asarray(nib.load(str(DERIV / sub / "anat" / f"{sub}_acq-{side}_US_dseg.nii.gz")).dataobj)

        us_r = _resample_to_spacing(us, us_sp, ct_sp, order=1)          # uint8, downsampled (small)
        gt_r = _resample_to_spacing((us_gt > 0).astype(np.uint8), us_sp, ct_sp, order=0)
        fan = _fan_mask(us_r)                            # full fan, annotations stripped
        # rescale the fan's US intensities into the CT HU range (float, small)
        us_hu = np.zeros(us_r.shape, np.float32)
        if fan.any():
            v = us_r[fan].astype(np.float32); mn, mx = v.min(), v.max()
            us_hu[fan] = lo + (v - mn) / max(1e-6, (mx - mn)) * (hi - lo)

        cs, bs = _paste(chim, us_r, cents[side])
        if cs is None:
            continue
        m = fan[bs]                                      # paste the full fan (no annotations)
        chim[cs][m] = us_hu[bs][m]
        # placed GT + ROI within the pasted footprint
        pg = placed_gt[cs]; pg[m & (gt_r[bs] > 0)] = 1; placed_gt[cs] = pg
        rr = roi[cs]; rr[m] = SIDE_ROI[side]; roi[cs] = rr
        inserted.append(side)

    if not inserted:
        return pid, "no side could be placed", []

    aff, hdr = ct_img.affine, ct_img.header
    NN.joinpath("imagesTs_chimera").mkdir(parents=True, exist_ok=True)
    NN.joinpath("labelsTs_chimera").mkdir(parents=True, exist_ok=True)
    NN.joinpath("roiTs_chimera").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(chim.astype(np.int16), aff, hdr), str(NN / "imagesTs_chimera" / f"{pid}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(placed_gt, aff, hdr), str(NN / "labelsTs_chimera" / f"{pid}.nii.gz"))
    nib.save(nib.Nifti1Image(roi, aff, hdr), str(NN / "roiTs_chimera" / f"{pid}.nii.gz"))
    return pid, "ok", inserted


def main():
    # patients with ≥1 US side, from the BIDS tree
    pats = {}
    for p in sorted(BIDS.glob("sub-*/anat")):
        pid = p.parent.name[len("sub-"):]
        sides = [s for s in ("R", "L") if (p / f"sub-{pid}_acq-{s}_US.nii.gz").exists()]
        if sides and (p / f"sub-{pid}_CT.nii.gz").exists():
            pats[pid] = sides
    # optional CLI filter: build only the given patient ids (for quick QA)
    only = set(sys.argv[1:])
    if only:
        pats = {k: v for k, v in pats.items() if k in only}
    print(f"Building chimeras for {len(pats)} patients (≥1 US side + CT) with {WORKERS} workers …")

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for pid, status, inserted in pool.map(build_patient, list(pats.items())):
            rows.append({"patient": pid, "status": status,
                         "sides_inserted": "".join(inserted), "n_sides": len(inserted)})
            print(f"  {pid}: {status}  sides={inserted}")

    man = NN / "chimera_manifest.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "status", "sides_inserted", "n_sides"])
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["patient"]))
    ok = sum(1 for r in rows if r["status"] == "ok")
    both = sum(1 for r in rows if r["n_sides"] == 2)
    print(f"\nDone: {ok}/{len(rows)} chimeras built ({both} with both kidneys). Manifest → {man}")


if __name__ == "__main__":
    main()
