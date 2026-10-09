#!/usr/bin/env python
"""
Level-1 texture metric, gradient-orientation variant: Normalized Gradient Field (NGF) similarity.

Per-voxel similarity between SOURCE and GENERATED gradients:

    ngf(voxel) = (grad_src . grad_gen)^2 / ((|grad_src|^2 + eps^2) * (|grad_gen|^2 + eps^2))

= squared cosine of the angle between the two local gradient vectors. This is the similarity
measure from Haber & Modersitzki 2006 ("Intensity gradient based registration and fusion of
multi-modal images", MICCAI) — proposed specifically for comparing images of DIFFERENT contrast
(multimodal registration), still current best practice (used in the 2020 Learn2Reg challenge,
Häger et al.). Formula verified against a maintained reference implementation:
https://github.com/BailiangJ/normalized_gradient_field_pytorch/blob/master/normalized_gradient_field.py

Why this instead of (or alongside) census/census_local8:
  * Contrast invariance is POINTWISE, not block-wise. By the chain rule, grad(g(x)) = g'(x)*grad(x)
    for ANY differentiable g — so gradient DIRECTION survives any locally-monotonic remap
    (increasing or decreasing; decreasing flips sign, which squaring absorbs) regardless of how
    nonlinear g is or how much it varies from region to region. No block-size hyperparameter is
    needed to avoid PALETTE's cross-region sign cancellation (see census_local8 in
    compute_texture_metrics_openms.py) — each voxel's gradient is already local (3-tap kernel), and
    each voxel's contribution is already non-negative (squared) before any spatial aggregation.
  * NOT equivalent to census_local8: this is not "make the block smaller", it changes WHAT is
    compared (gradient direction, i.e. edge/boundary orientation) rather than WHAT (rank order
    over a window). Reported alongside census metrics as an independent line of evidence.

Honest floor: two INDEPENDENT 3-D gradient vectors have E[cos^2(angle)] = 1/3 by symmetry (chance
alignment in 3 dims), NOT 0 — verified empirically below by --sanity, not assumed.

Edge gating: eps is for numerical stability only (not a tunable "edge sensitivity" knob — the
project's on-harmony NGF evaluation flagged exactly this as confusing). Instead we explicitly
report two aggregates:
  * ngf_all  = mean over ALL ROI voxels.
  * ngf_edge = mean restricted to the top 50% of ROI voxels by |grad_src| (i.e. "where the SOURCE
    actually has an edge") — a data-driven gate, not a tuned threshold, expressing "does the
    texture that exists survive" rather than being diluted by flat/uninformative voxels.

Usage:
  python compute_ngf_texture.py --sanity                      # metric self-test, no data
  python compute_ngf_texture.py --device cuda --source FLAIR --set-label flair_noblur \
      --generated-root .../data/generated_noblur --output-csv .../ngf_flair_noblur.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_texture_metrics_openms import (   # reuse I/O + ROI discovery, UNCHANGED
    list_source_keys, foreground_mask, _load, _load_lesion, erode,
    CONTRASTS, GENERATED_METHODS, CONTROL_METHODS, GAMMA_VALUES, make_control,
    DEFAULT_GENERATED, PROJECT_ROOT,
)

EPS = 1e-5          # numerical stability only, NOT an edge-sensitivity knob (see docstring)
EDGE_PCTL = 50       # ngf_edge = mean over top-(100-EDGE_PCTL)% |grad_src| voxels in the ROI


# ─────────────────────────── gradient (3-tap separable Sobel, 3-D) ──────────────────────────
def _sobel_kernels(device, dtype):
    """Separable 3x3x3 Sobel: derivative [1,0,-1] along the gradient axis, smoothing [1,2,1] along
    the other two. Standard first-derivative kernel (Sobel 1968); same family used by every NGF
    reference implementation. Returns 3 (1,1,3,3,3) kernels for (dz, dy, dx)."""
    d = torch.tensor([1., 0., -1.], device=device, dtype=dtype)
    s = torch.tensor([1., 2., 1.], device=device, dtype=dtype)

    def outer3(a, b, c):
        return torch.einsum('i,j,k->ijk', a, b, c)

    kz = outer3(d, s, s)[None, None]
    ky = outer3(s, d, s)[None, None]
    kx = outer3(s, s, d)[None, None]
    return kz, ky, kx


def gradient_3d(x: torch.Tensor):
    """x: (D,H,W). Returns (gz, gy, gx), each (D,H,W), reflect-padded so shape is preserved."""
    kz, ky, kx = _sobel_kernels(x.device, x.dtype)
    xp = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = F.conv3d(xp, kz)[0, 0]
    gy = F.conv3d(xp, ky)[0, 0]
    gx = F.conv3d(xp, kx)[0, 0]
    return gz, gy, gx


def ngf_map(src, gen, eps=EPS):
    """Per-voxel squared-cosine similarity between src and gen gradient vectors. src,gen: (D,H,W)."""
    sz, sy, sx = gradient_3d(src)
    gz, gy, gx = gradient_3d(gen)
    dot = sz * gz + sy * gy + sx * gx
    src_norm2 = sz * sz + sy * sy + sx * sx + eps ** 2
    gen_norm2 = gz * gz + gy * gy + gx * gx + eps ** 2
    return (dot * dot) / (src_norm2 * gen_norm2), (sz * sz + sy * sy + sx * sx)


def ngf_scores(src, gen, mask, edge_pctl=EDGE_PCTL, eps=EPS):
    """Returns (ngf_all, ngf_edge, n_vox) over `mask`. ngf_edge restricts to the top
    (100-edge_pctl)% of in-mask voxels by source gradient magnitude."""
    sim, src_g2 = ngf_map(src, gen, eps)
    v_sim, v_g2 = sim[mask], src_g2[mask]
    n = int(mask.sum().item())
    if n == 0:
        return float("nan"), float("nan"), 0
    ngf_all = float(v_sim.mean())
    thresh = torch.quantile(v_g2, edge_pctl / 100.0)
    edge_sel = v_g2 >= thresh
    ngf_edge = float(v_sim[edge_sel].mean()) if int(edge_sel.sum()) > 0 else float("nan")
    return ngf_all, ngf_edge, n


# ─────────────────────────── sanity self-test ───────────────────────────────
def run_sanity(device) -> int:
    torch.manual_seed(0)
    D = 44

    def boxf(x, w):
        return F.avg_pool3d(x[None, None], w, 1, w // 2, count_include_pad=False)[0, 0]

    source = boxf(torch.randn(D, D, D, device=device), 5)
    source = (source - source.min()) / (source.max() - source.min() + 1e-7)
    mask = torch.ones(D, D, D, dtype=torch.bool, device=device)
    mask[:3] = mask[-3:] = mask[:, :3] = mask[:, -3:] = mask[:, :, :3] = mask[:, :, -3:] = False  # drop pad edge

    def palette_like(x, nb=6):
        centers = torch.rand(nb, 3, device=device) * D
        coords = torch.stack(torch.meshgrid(*[torch.arange(D, device=device)] * 3, indexing="ij"), -1).float()
        lab = ((coords[..., None, :] - centers) ** 2).sum(-1).argmin(-1)
        out = x.clone()
        for b in range(nb):
            mb = lab == b
            if mb.sum() < 10:
                continue
            mu = torch.rand(1, device=device).item()
            alpha = (torch.rand(1, device=device).item() * 1.5 + 0.5) * (1 if torch.rand(1) > .5 else -1)
            out[mb] = mu + alpha * (x[mb] - x[mb].mean())
        return out

    palette_draws = [palette_like(source) for _ in range(4)]
    cases = {
        "identity": source.clone(),
        "gamma":    source.clamp_min(1e-4) ** 2.0,          # monotone-increasing
        "log":      torch.log1p(9 * source.clamp_min(0)),   # nonlinear monotone-increasing
        "inverted": 1.0 - source,                            # monotone-DEcreasing
        "noise":    torch.rand(D, D, D, device=device),
    }
    print(f"{'case':16s} {'ngf_all':>10s} {'ngf_edge':>10s}")
    res = {}
    for name, gen in cases.items():
        a, e, _ = ngf_scores(source, gen, mask)
        res[name] = (a, e)
        print(f"{name:16s} {a:10.3f} {e:10.3f}")
    pal = np.mean([ngf_scores(source, d, mask)[0] for d in palette_draws])
    pal_e = np.mean([ngf_scores(source, d, mask)[1] for d in palette_draws])
    print(f"{'palette-like(avg4)':16s} {pal:10.3f} {pal_e:10.3f}")

    checks = [
        ("identity ngf_all≈1",              res["identity"][0] > 0.95),
        ("gamma ngf_all≈1 (monotone incr)", res["gamma"][0]    > 0.95),
        ("log ngf_all≈1 (nonlinear monot)", res["log"][0]      > 0.95),
        ("inverted ngf_all≈1 (squared)",    res["inverted"][0] > 0.95),
        ("noise ngf_all≈1/3 chance floor",  0.20 < res["noise"][0] < 0.50),
        ("palette-like ≫ noise floor",      pal > res["noise"][0] + 0.10),
        ("edge-gated ≥ all (source-driven)", pal_e >= pal - 0.05),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"); ok &= passed
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ─────────────────────────── task enumeration (mirrors compute_texture_metrics_openms) ──────
def build_tasks(generated_root, source_keys, methods):
    tasks = []
    for method in methods:
        if method in CONTROL_METHODS:
            for key in source_keys:
                for v in (GAMMA_VALUES if method == "gamma" else [0]):
                    tasks.append((method, key, v, None))
            continue
        mdir = generated_root / method
        for key in source_keys:
            for synth in sorted((mdir / key).glob(f"{key}_run-*.nii.gz")):
                variant = int(synth.stem.split("run-")[-1].split(".")[0])
                tasks.append((method, key, variant, synth))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED)
    p.add_argument("--source", type=str, choices=list(CONTRASTS), required=False)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--methods", type=str, default=",".join(GENERATED_METHODS))
    p.add_argument("--set-label", type=str, default="flair")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--erode-iters", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.sanity:
        sys.exit(run_sanity(device))

    if not args.source:
        sys.exit("--source FLAIR|T1w is required (unless --sanity)")

    import pandas as pd
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    source_keys = list_source_keys(args.source)
    if not source_keys:
        log.error("No sources for contrast=%s", args.source); sys.exit(1)

    tasks = build_tasks(args.generated_root, source_keys, methods)
    tasks = tasks[args.rank::args.world_size]
    if args.limit:
        tasks = tasks[:args.limit]
    log.info("set=%s source=%s methods=%s rank=%d/%d tasks=%d",
             args.set_label, args.source, methods, args.rank, args.world_size, len(tasks))

    source_cache, mask_cache = {}, {}
    rows = []
    for i, (method, key, variant, synth_path) in enumerate(tasks):
        try:
            if key not in source_cache:
                img_path, les_path = source_keys[key]
                src = _load(img_path, device)
                lesion = _load_lesion(les_path, src.shape, device)
                fg = foreground_mask(src)
                source_cache[key] = src
                mask_cache[key] = {"lesion": lesion, "foreground": fg}
            source, masks = source_cache[key], mask_cache[key]

            if method in CONTROL_METHODS:
                synth = make_control(source, masks["foreground"], method, variant)
            else:
                synth = _load(synth_path, device)
                if synth.shape != source.shape:
                    continue

            for roi_name, base_mask in masks.items():
                m = erode(base_mask, args.erode_iters)
                if int(m.sum()) < 50:
                    continue
                ngf_all, ngf_edge, n = ngf_scores(source, synth, m)
                rows.append(dict(set=args.set_label, method=method, subject=key.split("_")[0],
                                 session="single", variant=variant, roi_id=roi_name, n_vox=n,
                                 ngf_all=ngf_all, ngf_edge=ngf_edge))
        except Exception as e:                                        # noqa: BLE001
            log.warning("FAILED %s/%s/%s: %s", method, key, variant, e)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d done", i + 1, len(tasks))

    df = pd.DataFrame(rows)
    out = args.output_csv or (THIS_DIR.parent / "outputs" / "data" / f"ngf_{args.set_label}_rank{args.rank}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %d rows → %s", len(df), out)
    if len(df):
        log.info("\n%s", df.groupby(["method", "roi_id"])[["ngf_all", "ngf_edge"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
