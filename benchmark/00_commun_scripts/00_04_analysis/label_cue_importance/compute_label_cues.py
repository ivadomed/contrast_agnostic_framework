#!/usr/bin/env python
"""
Compute the four boundary cues (bg / eg_coh / eg_sharp / tg) at the surface of each segmentation
label of each subject, plus matched non-boundary points, and write one .npz per (subject, label).

Shared across datasets -- a dataset contributes only a small YAML/CLI spec of where its
(image, label) pairs live. Metric definitions, grounding and honesty statements live in
`cue_metrics.py` and `LITERATURE_REVIEW.md` (same directory); this file is I/O and sampling only.

Sampling design (LITERATURE_REVIEW.md sec.5 -- the estimand depends on it):
  positives  voxels of label L whose 6-neighbourhood leaves L (the inner surface), subsampled.
  negatives  non-boundary points drawn 50/50 from L's INTERIOR and from its immediate
             SURROUNDINGS, never from far-away anatomy -- so the question is "what distinguishes
             the annotated surface from nearby non-surface locations", not the trivially easy
             "what distinguishes lesion from healthy brain".
  Negatives are kept >= BALL_RADIUS+1 voxels from ANY label surface: a "non-boundary" point closer
  than the analysis window still has the boundary inside its half-ball, so no window-based cue
  could separate it from a true boundary point. --sanity applies the identical exclusion.

Usage (via a dataset's 7_analysis_<ds>/label_cue_importance_lvl_1/scripts/run_*.sh, not by hand):
  python compute_label_cues.py --images DIR --labels DIR --out-dir DIR --dataset NAME \
      --modality t1n --labels-json path/to/dataset.json [--n-subjects 40] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_metrics import (BALL_RADIUS, CUE_NAMES, PROFILE_HALFLEN, compute_cues,  # noqa: E402
                         ngf_label_alignment, regional_texture_contrast)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_POS = 1500          # per (subject, label); caps cost, not statistics -- see aggregate script
SURROUND_DIST = 25      # negatives from the "surroundings" stratum come from within this distance
FG_PCTL_FRAC = 0.10     # foreground = intensity > 10% of p99, the project's existing convention


def _binary_dilate(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """6-connected binary dilation, `iters` times (chessboard-free, matches the surface definition)."""
    x = mask[None, None].float()
    k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
    k[0, 0, 1, 1, :] = 1; k[0, 0, 1, :, 1] = 1; k[0, 0, :, 1, 1] = 1
    for _ in range(iters):
        x = (F.conv3d(F.pad(x, (1,) * 6), k) > 0).float()
    return x[0, 0] > 0


def surface_voxels(lab: torch.Tensor) -> torch.Tensor:
    """Inner surface of a binary mask: voxels in the mask with a 6-neighbour outside it."""
    return lab & ~_erode6(lab)


def _erode6(mask: torch.Tensor) -> torch.Tensor:
    x = mask[None, None].float()
    k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
    k[0, 0, 1, 1, :] = 1; k[0, 0, 1, :, 1] = 1; k[0, 0, :, 1, 1] = 1
    n = int(k.sum().item())
    return (F.conv3d(F.pad(x, (1,) * 6, mode="replicate"), k)[0, 0] >= n - 0.5)


def _sample(mask: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    idx = mask.nonzero()
    if idx.shape[0] == 0:
        return idx
    if idx.shape[0] > k:
        idx = idx[torch.randperm(idx.shape[0], generator=gen, device=idx.device)[:k]]
    return idx


def points_for_label(lab_all: torch.Tensor, target: int, fg: torch.Tensor, shape,
                     gen: torch.Generator):
    """-> (points (N,3) long, y (N,) float, stratum (N,) int) ; strata: 0=surface 1=interior 2=outside."""
    dev = lab_all.device
    lab = lab_all == target
    if int(lab.sum()) == 0:
        return None
    surf = surface_voxels(lab)

    margin = BALL_RADIUS + PROFILE_HALFLEN + 1          # keep every analysis window in-volume
    inside = torch.zeros(shape, dtype=torch.bool, device=dev)
    inside[margin:shape[0] - margin, margin:shape[1] - margin, margin:shape[2] - margin] = True

    # Any label's surface, dilated by the window radius -> the zone negatives may NOT come from.
    any_surface = torch.zeros(shape, dtype=torch.bool, device=dev)
    for v in torch.unique(lab_all):
        if int(v) != 0:
            any_surface |= surface_voxels(lab_all == v)
    forbidden = _binary_dilate(any_surface, BALL_RADIUS + 1)

    pos = _sample(surf & inside & fg, MAX_POS, gen)
    if pos.shape[0] < 30:                                # too small to score; caller skips
        return None
    n_each = max(1, pos.shape[0] // 2)
    interior = _sample(lab & ~forbidden & inside & fg, n_each, gen)
    surround = _sample(_binary_dilate(lab, SURROUND_DIST) & ~lab & ~forbidden & inside & fg,
                       n_each, gen)
    if interior.shape[0] + surround.shape[0] < 30:
        return None

    # Balance positives against negatives so average precision has a chance baseline of exactly
    # 0.5 for every label and dataset. Without this the baseline is the positive prevalence, which
    # varies with label size -- a smoke run gave 250 pos vs 125 neg, i.e. a 0.667 chance line that
    # sat ABOVE every measured cue, which would have read as "all cues informative" when in fact
    # all four were at or below chance. Not optional: the whole table is cross-label comparisons.
    n_neg = interior.shape[0] + surround.shape[0]
    if pos.shape[0] > n_neg:
        pos = pos[torch.randperm(pos.shape[0], generator=gen, device=pos.device)[:n_neg]]

    pts = torch.cat([pos, interior, surround], 0)
    y = np.r_[np.ones(len(pos)), np.zeros(len(interior) + len(surround))]
    strat = np.r_[np.zeros(len(pos)), np.ones(len(interior)), 2 * np.ones(len(surround))]
    return pts, y, strat.astype(np.int8)


TARGET_MM = 1.0   # every volume is resampled to this ISOTROPIC spacing before any cue is computed


def load_volume(path: Path, device, is_label: bool = False):
    """Load a volume and resample it to TARGET_MM isotropic. Returns (tensor, original_zooms).

    THIS IS NOT OPTIONAL, and it is the single most important preprocessing step in the analysis.
    Every radius in cue_metrics.py -- the half-ball, the texture radii, the profile half-length,
    the negative-sampling exclusion -- is expressed in VOXELS. The datasets compared here differ in
    anisotropy by up to 6.6x:

        chaos    in-plane 1.36-1.89 mm, SLICE 5.5-9.0 mm   (2.9x - 6.6x anisotropic)
        brats    1.0 mm isotropic
        open-ms  1.0 mm isotropic

    So without resampling, a "ball of radius 7" is a true 7 mm sphere in brats/open-ms but a
    ~10 mm x ~10 mm x ~50-63 mm CIGAR in chaos, spanning ten slices of unrelated anatomy. That
    destroys precisely the gradient-direction coherence eg_coh is meant to measure (a liver edge is
    an obvious wall, yet scored only 0.639), mis-scales the through-slice Sobel derivative by 4-6x,
    and puts eg_sharp's +-4-voxel profile at +-36 mm. It also biases the headline cross-dataset
    comparison, handicapping the anisotropic dataset only -- i.e. it made the boundary gap look
    SMALLER than it is. Found 2026-08-02 after the full run, when the chaos wall came out far less
    decisive than the anatomy obviously warrants.

    Labels use nearest-neighbour; images trilinear."""
    img = nib.load(str(path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    t = torch.from_numpy(arr).to(device)
    if max(abs(z - TARGET_MM) for z in zooms) < 1e-3:
        return t, zooms
    scale = [z / TARGET_MM for z in zooms]
    out_shape = [max(1, int(round(sh * sc))) for sh, sc in zip(t.shape, scale)]
    mode = "nearest" if is_label else "trilinear"
    kw = {} if is_label else {"align_corners": False}
    t = F.interpolate(t[None, None], size=out_shape, mode=mode, **kw)[0, 0]
    return t, zooms


def foreground(vol: torch.Tensor) -> torch.Tensor:
    """intensity > 10% of p99 -- the same foreground rule as the project's Pillar-1 texture
    analysis, so the two analyses see the same voxels."""
    p99 = torch.quantile(vol[vol > 0].float()[:1_000_000], 0.99) if (vol > 0).any() else vol.max()
    return vol > FG_PCTL_FRAC * p99


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="dir of <case><image-suffix>")
    ap.add_argument("--labels", required=True, type=Path, help="dir of <case><label-suffix>")
    # Suffixes are options because BraTS t1c/t2f come from the BIDS tree (uncompressed .nii,
    # staged as flat symlinks by the dataset wrapper) while every other branch reads nnUNet raw.
    ap.add_argument("--image-suffix", default="_0000.nii.gz")
    ap.add_argument("--label-suffix", default=".nii.gz")
    ap.add_argument("--labels-json", required=True, type=Path, help="nnUNet dataset.json (label map)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--extra-union", default="", help="comma-separated label ids to ALSO score as "
                                                      "one merged region, e.g. BraTS whole tumour")
    ap.add_argument("--union-name", default="union")
    args = ap.parse_args()

    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_map = {v: k for k, v in json.loads(args.labels_json.read_text())["labels"].items()
                 if int(v) != 0}
    union_ids = [int(x) for x in args.extra_union.split(",") if x.strip()]

    cases = sorted(p.name[:-len(args.image_suffix)]
                   for p in args.images.glob(f"*{args.image_suffix}"))
    if not cases:
        log.error("no *%s under %s", args.image_suffix, args.images)
        return 1
    rng = np.random.default_rng(args.seed)
    if len(cases) > args.n_subjects:
        cases = sorted(rng.choice(cases, args.n_subjects, replace=False).tolist())
    log.info("%s/%s: %d subjects, labels=%s%s", args.dataset, args.modality, len(cases),
             label_map, f", union{union_ids}" if union_ids else "")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    n_written = 0
    for i, case in enumerate(cases, 1):
        ipath = args.images / f"{case}{args.image_suffix}"
        lpath = args.labels / f"{case}{args.label_suffix}"
        if not lpath.exists():
            log.warning("[%d/%d] %s: no label, skipped", i, len(cases), case)
            continue
        vol, zooms = load_volume(ipath, dev)
        lab_all, _ = load_volume(lpath, dev, is_label=True)
        lab_all = lab_all.round().to(torch.int64)
        if vol.shape != lab_all.shape:
            log.warning("[%d/%d] %s: shape mismatch %s vs %s, skipped", i, len(cases), case,
                        vol.shape, lab_all.shape)
            continue
        if i == 1:
            log.info("spacing %s -> %.1fmm isotropic, shape %s", tuple(round(z, 2) for z in zooms),
                     TARGET_MM, tuple(vol.shape))
        fg = foreground(vol)
        vol = (vol - vol[fg].mean()) / vol[fg].std().clamp_min(1e-6)

        targets = [(int(v), n) for v, n in label_map.items()]
        if union_ids:
            lab_all = torch.where(torch.isin(lab_all, torch.tensor(union_ids, device=dev)),
                                  lab_all, lab_all)   # unchanged; union handled as a virtual label
        for value, name in targets + ([(-1, args.union_name)] if union_ids else []):
            if value == -1:
                lab_src = torch.isin(lab_all, torch.tensor(union_ids, device=dev)).to(torch.int64)
                got = points_for_label(lab_src, 1, fg, vol.shape, gen)
                binary = lab_src == 1
            else:
                got = points_for_label(lab_all, value, fg, vol.shape, gen)
                binary = lab_all == value
            if got is None:
                continue
            pts, y, strat = got
            cues = compute_cues(vol, fg, pts)
            # Descriptive NGF(image, label) surface-alignment index -- the ablation's own NGF
            # formula, reported against a spatial permutation null. NOT a detector cue (that use
            # is circular); see ngf_label_alignment's docstring.
            ngf = ngf_label_alignment(vol, binary, fg, gen=gen)
            # Regional texture contrast: ROI interior vs its surrounding shell, multi-scale HOG,
            # against a placement null. Answers "does this region's texture differ from the tissue
            # around it" -- a different question from the half-ball cues' "is there a texture STEP
            # exactly at the outline", and at a scale a radius-7 ball cannot reach.
            reg = regional_texture_contrast(vol, binary, fg, gen=gen)
            out = args.out_dir / f"{args.dataset}__{args.modality}__{case}__{name}.npz"
            np.savez_compressed(out, y=y, stratum=strat,
                                **{f"ngf_{k}": np.float64(v) for k, v in ngf.items()},
                                **{f"reg_{k}": np.float64(v) for k, v in reg.items()},
                                **{k: v.detach().cpu().numpy() for k, v in cues.items()})
            n_written += 1
        log.info("[%d/%d] %s done", i, len(cases), case)

    log.info("wrote %d (subject,label) files to %s", n_written, args.out_dir)
    return 0 if n_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
