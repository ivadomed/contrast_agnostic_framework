#!/usr/bin/env python
"""
Diagnostic for ANOMALY 2: why does whole-ROI census rank auglab_default > palette, when
palette (real-intensity affine remap) should preserve MORE texture than auglab's mixed
global augmentations?

Hypothesis: the whole-ROI census |corr| is confounded by PALETTE's SIGNED per-region affine
remap. PALETTE fills each K-means/Voronoi region with  y = mu_c + alpha_c*(x - mean_c),
alpha_c signed (fromSeg.py:444-446 etc.) — so within a region the local rank field is either
preserved (alpha>0) or inverted (alpha<0). abscorr takes |corr| over the WHOLE ROI at once,
so +corr regions and -corr regions CANCEL in the global sum before |.| -> |global corr|
collapses toward 0 even though |within-region corr| ~ 1. auglab's dominant transforms
(Gamma/Contrast/Inverse/HistEq) are GLOBAL monotone remaps (one sign everywhere) -> no
cancellation -> high global census.

Test: block-wise LOCAL census. Tile the ROI into small cubes (mostly single-region), compute
|corr| per block, average. This removes cross-region sign cancellation. Prediction:
  palette -> near ceiling (within-region texture preserved),
  auglab  -> moderate (genuine local texture alteration by RandomConv/Scharr/noise),
  synthseg_* / baseline_kmeans_label_remap_voronoi -> floor (noise fill has no local texture).
i.e. palette > auglab > floor, matching design intent.

Runs on the FINAL generated_noblur volumes (FLAIR source, all 30 subjects, run-00).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import nibabel as nib

REPO = Path("/project/6102268/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts"))
from compute_texture_metrics_openms import (
    rank_transform, abscorr, list_source_keys, foreground_mask, _load_lesion,
)

GENERATED = REPO / "datasets/open-ms/7_analysis_open-ms/data/generated_noblur"
METHODS = ["palette", "auglab_default", "synthseg_em", "synthseg_noem", "baseline_kmeans_label_remap_voronoi"]
BLOCK_SIZES = [8, 16]        # cube edge (voxels)
MIN_BLOCK_VOX = 64           # a block needs this many in-ROI voxels to count
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    arr = np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return torch.from_numpy(arr).to(device)


def global_census(sr, yr, mask):
    return abscorr(sr, yr, mask)


def local_census(sr, yr, mask, block):
    """Mean over blocks of |corr(sr, yr)| within (block ∩ mask); blocks with <MIN_BLOCK_VOX skipped.
    Also returns the mean SIGNED corr magnitude collapse ratio to expose cancellation."""
    D, H, W = sr.shape
    vals, weights = [], []
    signed_vals = []
    for z in range(0, D, block):
        for y in range(0, H, block):
            for x in range(0, W, block):
                m = mask[z:z+block, y:y+block, x:x+block]
                n = int(m.sum().item())
                if n < MIN_BLOCK_VOX:
                    continue
                a = sr[z:z+block, y:y+block, x:x+block][m]
                b = yr[z:z+block, y:y+block, x:x+block][m]
                a = a - a.mean(); b = b - b.mean()
                denom = a.norm() * b.norm()
                if denom < 1e-7:
                    continue
                c = float((a * b).sum() / denom)      # SIGNED per-block corr
                vals.append(abs(c)); signed_vals.append(c); weights.append(n)
    if not vals:
        return float("nan"), float("nan"), 0
    w = np.array(weights, float)
    mean_abs = float(np.average(vals, weights=w))       # cancellation-robust local census
    mean_signed_abs = abs(float(np.average(signed_vals, weights=w)))  # |mean signed| — low if signs mix
    return mean_abs, mean_signed_abs, len(vals)


source_keys = list_source_keys("FLAIR")
print(f"{len(source_keys)} FLAIR subjects; block sizes {BLOCK_SIZES}\n", flush=True)

agg = {m: {"global": {"lesion": [], "fg": []},
           **{f"local{b}": {"lesion": [], "fg": []} for b in BLOCK_SIZES},
           **{f"signcollapse{b}": {"lesion": [], "fg": []} for b in BLOCK_SIZES}}
       for m in METHODS}

for i, (key, (img_path, les_path)) in enumerate(sorted(source_keys.items())):
    source = load(img_path)
    lesion = _load_lesion(les_path, source.shape, device)
    fg = foreground_mask(source)
    sr = rank_transform(source, 1)
    rois = {"lesion": lesion, "fg": fg}

    for m in METHODS:
        runs = sorted((GENERATED / m / key).glob(f"{key}_run-*.nii.gz"))
        if not runs:
            continue
        synth = load(runs[0])
        if synth.shape != source.shape:
            continue
        yr = rank_transform(synth, 1)
        for rn, mask in rois.items():
            if int(mask.sum()) < 200:
                continue
            agg[m]["global"][rn].append(global_census(sr, yr, mask))
            for b in BLOCK_SIZES:
                la, sc, _ = local_census(sr, yr, mask, b)
                agg[m][f"local{b}"][rn].append(la)
                agg[m][f"signcollapse{b}"][rn].append(sc)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(source_keys)}", flush=True)


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


print("\n=== census_r1: whole-ROI (current metric) vs block-wise LOCAL (cancellation-robust) ===")
hdr = f"{'method':22s} {'ROI':10s} {'global':>8s}"
for b in BLOCK_SIZES:
    hdr += f" {'local'+str(b):>8s}"
for b in BLOCK_SIZES:
    hdr += f" {'|signed|'+str(b):>10s}"
print(hdr)
for m in METHODS:
    for rn in ("lesion", "fg"):
        row = f"{m:22s} {rn:10s} {mean(agg[m]['global'][rn]):8.3f}"
        for b in BLOCK_SIZES:
            row += f" {mean(agg[m][f'local{b}'][rn]):8.3f}"
        for b in BLOCK_SIZES:
            row += f" {mean(agg[m][f'signcollapse{b}'][rn]):10.3f}"
        print(row)
print("\nglobal  = |corr| over whole ROI (current headline metric)")
print("localB  = mean over BxBxB blocks of |corr| within block  (removes cross-region sign cancellation)")
print("|signed|B = |mean over blocks of SIGNED corr| — much lower than localB ⇒ per-region sign flips")
print("           (the cancellation fingerprint; expected large for palette, small for auglab).")
