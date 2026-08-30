#!/usr/bin/env python3
"""
Generates the 10 PALETTE pipeline-figure panels from scratch by running the
REAL transform code (src/synthesis/v26_6_synthesis.py, v26_6_2_synthesis.py)
on one real CHAOS T2SPIR slice and one real OnHarmony T1w slice -- not by
cropping a pre-existing composite image (the earlier, rejected approach).

Each 2-D slice is treated as a (1, 1, 1, H, W) pseudo-3D volume so the exact
same K-means / Voronoi / affine-remap helper functions used at training time
apply unmodified (D=1 is a valid, if degenerate, spatial dimension for the
Voronoi coordinate distance computation).

For illustration only, Voronoi sub-parcellation is forced on for every
cluster (force_split=True) and the label-remap step is applied to every
label (rather than the stochastic 50%/cluster-skip probabilities used during
actual training) so the panel sequence is fully visible in one deterministic
draw; this matches how the original (superseded) figure was framed as "one
representative pipeline pass," not a literal training sample.

Panels:
  (a) input
  (b) k-means flat remap: each cluster painted with its DOMINANT (largest)
      sub-region's target mean mu -- a flat preview before Voronoi splitting
      is visually applied.
  (c) + Voronoi sub-parcellation flat remap: each sub-region painted with its
      own mu (no texture).
  (d) + label flat remap: real anatomical labels additionally painted with
      their own mu, overwriting (c) on each label's footprint.
  (e) final output: the real signed-affine remap of Eq. (1) with texture
      (not a flat fill), at both the region and label level, matching
      v26_6_2_synthesis.py's actual per-label re-remap formula.

Usage:
  .venv/bin/python generate_method_figure_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.synthesis.v26_6_synthesis import (  # noqa: E402
    C_CHOICES,
    DARK_THRESHOLD,
    N_KMEANS_SUBSAMPLE,
    _kmeans_1d,
    _voronoi_region_ids,
)

OUT_DIR = REPO / "paper/cvpr_format_latex/figures/method_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)  # reproducible illustrative draw


def load_slice(img_path: Path, lbl_path: Path, axis: int = 2):
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)
    counts = (lbl > 0).reshape(-1, lbl.shape[axis]).sum(axis=0) if axis == lbl.ndim - 1 else None
    counts = [(lbl.take(i, axis=axis) > 0).sum() for i in range(lbl.shape[axis])]
    best = int(np.argmax(counts))
    img_slice = img.take(best, axis=axis)
    lbl_slice = lbl.take(best, axis=axis)
    return img_slice, lbl_slice


def normalize01(img_slice: np.ndarray) -> np.ndarray:
    fg = img_slice > np.percentile(img_slice, 1)
    lo, hi = img_slice[fg].min(), img_slice[fg].max()
    return np.clip((img_slice - lo) / (hi - lo + 1e-7), 0, 1)


def crop_to_foreground(img_slice: np.ndarray, lbl_slice: np.ndarray, pad: int = 8):
    """Tight bounding-box crop around the body/anatomy so panels aren't
    dominated by empty background margin (display-only; does not affect the
    transform itself, which already ran on the full slice)."""
    fg = img_slice > np.percentile(img_slice, 3)
    ys, xs = np.where(fg)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img_slice.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img_slice.shape[1])
    return img_slice[y0:y1, x0:x1], lbl_slice[y0:y1, x0:x1]


def run_pipeline(img_slice: np.ndarray, lbl_slice: np.ndarray, device="cpu"):
    H, W = img_slice.shape
    N = H * W
    flat = torch.from_numpy(img_slice.reshape(-1)).float().to(device)
    flat_m = (flat > DARK_THRESHOLD).float()
    lbl_flat = torch.from_numpy(lbl_slice.reshape(-1)).long().to(device)

    # ---- Step 1: k-means (1-D, on foreground intensities) ----
    C = C_CHOICES[int(torch.rand(1).item() * len(C_CHOICES))]
    fg_vals = flat[flat > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
    centroids = _kmeans_1d(fg_vals, C)
    sorted_c, sort_idx = torch.sort(centroids)
    boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
    lbl_s = torch.bucketize(flat, boundaries)
    lbl_l = sort_idx[lbl_s].long()

    # ---- Step 2: Voronoi sub-parcellation (forced on, for illustration) ----
    coords = torch.stack(torch.meshgrid(
        torch.arange(1, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), indexing="ij"), dim=-1).reshape(N, 3)
    rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C, torch.device(device), force_split=True)

    # ---- Step 3: per-region target stats (mu_c, alpha_c) + real mean ----
    eps = 1e-7
    s_c = torch.zeros(R).scatter_add_(0, rid, flat * flat_m)
    n_c = torch.zeros(R).scatter_add_(0, rid, flat_m)
    mean_c = s_c / n_c.clamp(min=eps)
    mu_c = torch.rand(R)
    mag_c = torch.rand(R) * 1.5 + 0.5
    sign_c = (torch.rand(R) > 0.5).float() * 2 - 1
    alpha_c = mag_c * sign_c

    # panel (b): k-means-only flat preview -- each CLUSTER painted with its
    # dominant (largest) sub-region's mu.
    region_size = n_c.clone()
    dominant_mu_per_cluster = torch.zeros(C)
    # map each region back to its parent cluster via lbl_l/rid correspondence
    region_to_cluster = torch.zeros(R, dtype=torch.long)
    seen = torch.zeros(R, dtype=torch.bool)
    rid_np, lbl_l_np = rid.numpy(), lbl_l.numpy()
    for c in range(C):
        c_regions = np.unique(rid_np[lbl_l_np == c])
        if len(c_regions) == 0:
            continue
        sizes = [n_c[r].item() for r in c_regions]
        dominant_mu_per_cluster[c] = mu_c[c_regions[int(np.argmax(sizes))]]
    panel_b = dominant_mu_per_cluster[lbl_l] * flat_m

    # panel (c): full sub-region flat fill (no texture)
    panel_c = mu_c[rid] * flat_m

    # ---- Step 4: per-label flat preview + real per-label texture remap ----
    unique_labels = [int(x) for x in torch.unique(lbl_flat) if x > 0]
    panel_d = panel_c.clone()
    step3_texture = (mu_c[rid] + alpha_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m
    panel_e = step3_texture.clone()
    for ell in unique_labels:
        ell_mask = (lbl_flat == ell).float()
        if ell_mask.sum() < 4:
            continue
        mu_ell = torch.rand(1).item()
        mag_ell = torch.rand(1).item() * 1.5 + 0.5
        sign_ell = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        alpha_ell = mag_ell * sign_ell
        panel_d = torch.where(ell_mask.bool(), torch.full_like(panel_d, mu_ell), panel_d)
        mean_ell_in_step3 = (step3_texture * ell_mask).sum() / ell_mask.sum().clamp(min=eps)
        remapped = (mu_ell + alpha_ell * (step3_texture - mean_ell_in_step3)).clamp(0, 1)
        panel_e = torch.where(ell_mask.bool(), remapped, panel_e)

    panels = {
        "a": flat, "b": panel_b, "c": panel_c, "d": panel_d, "e": panel_e.clamp(0, 1),
    }
    return {k: v.reshape(H, W).numpy() for k, v in panels.items()}


def save_panel(arr: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


CASES = [
    ("chaos", "datasets/chaos/2_nnUNet_chaos/raw/Dataset061_CHAOS_MR_T2spir/imagesTr/MR01_0000.nii.gz",
     "datasets/chaos/2_nnUNet_chaos/raw/Dataset061_CHAOS_MR_T2spir/labelsTr/MR01.nii.gz"),
    ("onharmony", "datasets/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/imagesTr/sub-03286_ses-NOT2ING001_T1w_0000.nii.gz",
     "datasets/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/labelsTr/sub-03286_ses-NOT2ING001_T1w.nii.gz"),
]

for slug, img_rel, lbl_rel in CASES:
    img_slice, lbl_slice = load_slice(REPO / img_rel, REPO / lbl_rel)
    img_slice, lbl_slice = crop_to_foreground(img_slice, lbl_slice)
    img01 = normalize01(img_slice)
    panels = run_pipeline(img01, lbl_slice)
    for letter, arr in panels.items():
        out = OUT_DIR / f"{slug}_{letter}.png"
        save_panel(arr, out)
        print(f"-> {out}")
