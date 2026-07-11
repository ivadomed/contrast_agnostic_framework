#!/usr/bin/env python
"""
Level-1 texture / structure-preservation metrics (input-space, network-free).

For each generated volume and each of the 31 anatomical ROIs, measure how much of the SOURCE
T1w's texture survives in the augmented volume, using two contrast-AND-inversion-invariant,
citable metrics (see datasets/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/LITERATURE_REVIEW.md
— moved there since texture analysis is now run on open-ms):

  * CENSUS — |corr( rank(source), rank(synth) )| within the eroded ROI.  PRIMARY texture metric.
             rank = census/rank transform (Zabih & Woodfill 1994; LBP ordinal family, Ojala
             2002): each voxel → fraction of neighbours it exceeds. Invariant to monotone remaps
             by construction; |·| handles PALETTE's α<0 inversion. Floor = 0 (independent rank
             fields → 0 correlation). Reported across window radii r ∈ {1,2} (columns
             census_r1 / census_r2) for robustness; r=1 is the stringent headline.
  * NMI    — Studholme normalized mutual information (H(X)+H(Y))/H(X,Y) ∈ [1,2]. Secondary
             content-preservation (Maes 1997 / Studholme 1999).

Read RELATIVELY: image-driven augs (PALETTE, auglab_default, gamma) sit far above the 0 floor;
generative SynthSeg sits on it. Absolutes are moderate (census is locally stringent).

Inline positive controls scored from the source (no volumes needed): `gamma`, `histeq` — pure
monotone remaps, so census → 1.0 (anchors the "perfect preservation" top and guards against a
rigged metric).

`--set-label` tags rows (blur / noblur / ref) so both generated sets combine into one table.
Volumes must be voxel-aligned to their source (guaranteed by generation).

Usage
-----
  python compute_texture_metrics.py --sanity                      # metric self-test, no data
  python compute_texture_metrics.py --device cuda --rank 0 --world-size 4 \
      --generated-root .../data/generated --set-label blur --output-csv .../metrics_blur_rank0.csv
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

THIS_DIR      = Path(__file__).resolve().parent
ANALYSIS_ROOT = THIS_DIR.parent                                  # texture_analysis_lvl_1/
# parents[0]=lvl1, [1]=7_analysis, [2]=on-harmony, [3]=datasets, [4]=repo root.
PROJECT_ROOT  = THIS_DIR.parents[4]

DEFAULT_GENERATED = ANALYSIS_ROOT / "data" / "generated"
DEFAULT_DS        = PROJECT_ROOT / "datasets" / "on-harmony" / "2_nnUNet_on-harmony" / "raw" / "Dataset031_OnHarmonyT1w31"
DEFAULT_IMAGES    = DEFAULT_DS / "imagesTr"
DEFAULT_LABELS    = DEFAULT_DS / "labelsTr"
DEFAULT_OUT       = ANALYSIS_ROOT / "outputs" / "data" / "texture_metrics.csv"

GENERATED_METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
CONTROL_METHODS   = ["gamma", "histeq"]          # synthesised inline from source
GAMMA_VALUES      = [0.5, 1.5, 2.5]

MIN_VOX    = 50        # ROI smaller than this (after erosion) → skip
EPS        = 1e-7
RANK_RADII = [1, 2]    # census window radii reported (robustness grid)


# ─────────────────────────── metric core (torch) ────────────────────────────
def rank_transform(x: torch.Tensor, r: int) -> torch.Tensor:
    """Census/rank transform: fraction of (2r+1)^3−1 neighbours that x exceeds. x,→ (D,H,W).
    Strict `>` (ties not counted) so flat regions → near-constant field → ~0 correlation."""
    out = torch.zeros_like(x)
    cnt = 0
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dz == dy == dx == 0:
                    continue
                out += (x > torch.roll(x, (dz, dy, dx), (0, 1, 2))).float()
                cnt += 1
    return out / cnt


def abscorr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
    """|Pearson correlation| of a,b over boolean mask m. |·| → inversion-invariant."""
    av, bv = a[m], b[m]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = av.norm() * bv.norm()
    if denom < EPS:
        return float("nan")
    return abs(float((av * bv).sum() / denom))


def nmi_1d(xv: torch.Tensor, yv: torch.Tensor, bins: int) -> float:
    """Studholme NMI = (H(X)+H(Y))/H(X,Y) on ROI voxel values xv, yv (1-D)."""
    def quant(v):
        vmin, vmax = v.min(), v.max()
        if (vmax - vmin) < EPS:
            return None
        return ((v - vmin) / (vmax - vmin) * (bins - 1)).round().long().clamp(0, bins - 1)

    xi, yi = quant(xv), quant(yv)
    if xi is None or yi is None:
        return float("nan")
    joint = torch.bincount(xi * bins + yi, minlength=bins * bins).float().reshape(bins, bins)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    hx  = -(px[px > 0] * px[px > 0].log()).sum()
    hy  = -(py[py > 0] * py[py > 0].log()).sum()
    hxy = -(joint[joint > 0] * joint[joint > 0].log()).sum()
    if hxy < EPS:
        return float("nan")
    return float(((hx + hy) / hxy).item())


def erode(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """Binary erosion via min-pool. mask: (D,H,W) bool → bool."""
    if iters <= 0:
        return mask
    m = mask.float()[None, None]
    for _ in range(iters):
        m = -F.max_pool3d(-m, kernel_size=3, stride=1, padding=1)
    return m[0, 0] > 0.5


# ─────────────────────────── per-volume computation ─────────────────────────
def metrics_for_pair(source, synth, labels, bins, erode_iters, src_ranks=None):
    """source/synth/labels: (D,H,W) on device. src_ranks: {r: rank_transform(source,r)} cached
    across a source's variants (computed here if None). Returns one row-dict per ROI."""
    if src_ranks is None:
        src_ranks = {r: rank_transform(source, r) for r in RANK_RADII}
    syn_ranks = {r: rank_transform(synth, r) for r in RANK_RADII}
    rows = []
    for roi in range(1, 32):                                           # 31 classes
        m = erode(labels == roi, erode_iters)
        n = int(m.sum().item())
        if n < MIN_VOX:
            continue
        row = dict(roi_id=roi, n_vox=n, nmi=nmi_1d(source[m], synth[m], bins))
        for r in RANK_RADII:
            row[f"census_r{r}"] = abscorr(src_ranks[r], syn_ranks[r], m)
        rows.append(row)
    return rows


# ─────────────────────────── I/O helpers ────────────────────────────────────
def _load(path: Path, device) -> torch.Tensor:
    import nibabel as nib
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def _foreground_norm01(x, fg):
    vals = x[fg]
    lo, hi = torch.quantile(vals, 0.01), torch.quantile(vals, 0.99)
    return ((x - lo) / (hi - lo + EPS)).clamp(0, 1)


def make_control(source, labels, kind, param):
    """Inline positive control synthesised from the source (monotone remap → census→1)."""
    fg = labels > 0
    s01 = _foreground_norm01(source, fg)
    if kind == "gamma":
        return s01.clamp_min(0) ** float(param)
    if kind == "histeq":
        from skimage.exposure import equalize_hist
        out = equalize_hist(s01.cpu().numpy(), mask=fg.cpu().numpy())
        return torch.from_numpy(out.astype(np.float32)).to(source.device)
    raise ValueError(kind)


# ─────────────────────────── task enumeration ───────────────────────────────
def list_source_keys(images_dir: Path):
    return {p.name[:-len("_0000.nii.gz")]: p
            for p in sorted(images_dir.glob("*_0000.nii.gz"))}   # sub-XXXX_ses-YYYY_T1w


def build_tasks(generated_root: Path, source_keys: dict, methods):
    """Each task = (method, key, variant, synth_path|None)."""
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


# ─────────────────────────── sanity self-test ───────────────────────────────
def run_sanity(device, bins, erode_iters) -> int:
    torch.manual_seed(0)
    D = 44

    def boxf(x, w):
        return F.avg_pool3d(x[None, None], w, 1, w // 2, count_include_pad=False)[0, 0]

    source = boxf(torch.randn(D, D, D, device=device), 5)
    source = (source - source.min()) / (source.max() - source.min() + EPS)
    labels = torch.zeros(D, D, D, dtype=torch.long, device=device)
    labels[5:39, 5:39, 5:22] = 1
    labels[5:39, 5:39, 22:39] = 2

    def palette_like(x, nb=6):
        centers = torch.rand(nb, 3, device=device) * D
        coords  = torch.stack(torch.meshgrid(*[torch.arange(D, device=device)] * 3, indexing="ij"), -1).float()
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

    # palette-like is high-variance on a phantom (few random blocks) — average several draws so
    # the sanity tests the METRIC (piecewise-monotone remap separates from noise), not a fragile
    # single-draw absolute. Real PALETTE numbers come from real volumes, not this.
    palette_draws = [palette_like(source) for _ in range(4)]
    cases = {
        "identity":     source.clone(),
        "gamma":        source.clamp_min(0) ** 2.0,
        "inverted":     1.0 - source,
        "noise":        torch.rand(D, D, D, device=device),
    }
    print(f"{'case':16s} {'census_r1':>10s} {'census_r2':>10s} {'NMI':>8s}")
    res = {}
    for name, synth in cases.items():
        rows = metrics_for_pair(source, synth, labels, bins, erode_iters)
        res[name] = (float(np.nanmean([r["census_r1"] for r in rows])),
                     float(np.nanmean([r["census_r2"] for r in rows])),
                     float(np.nanmean([r["nmi"] for r in rows])))
        print(f"{name:16s} {res[name][0]:10.3f} {res[name][1]:10.3f} {res[name][2]:8.3f}")
    pal = float(np.mean([np.nanmean([r["census_r1"] for r in
                 metrics_for_pair(source, d, labels, bins, erode_iters)]) for d in palette_draws]))
    print(f"{'palette-like(avg4)':16s} {pal:10.3f}")

    checks = [
        ("identity census≈1",         res["identity"][0] > 0.95),
        ("gamma census≈1 (monotone)", res["gamma"][0]    > 0.95),
        ("inverted census≈1 (|·|)",   res["inverted"][0] > 0.95),
        ("noise census≈0 floor",      res["noise"][0]    < 0.15),
        ("palette-like ≫ noise floor", pal > res["noise"][0] + 0.15),
        ("identity NMI≈2",            res["identity"][2] > 1.90),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"); ok &= passed
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ─────────────────────────── main ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--methods", type=str, default=",".join(GENERATED_METHODS))
    p.add_argument("--set-label", type=str, default="blur", help="tag rows: blur / noblur / ref")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mi-bins", type=int, default=64)
    p.add_argument("--erode-iters", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.sanity:
        sys.exit(run_sanity(device, args.mi_bins, args.erode_iters))

    import pandas as pd
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    source_keys = list_source_keys(args.images_dir)
    if not source_keys:
        log.error("No source images in %s", args.images_dir); sys.exit(1)
    log.info("set=%s  methods=%s  sources=%d", args.set_label, methods, len(source_keys))

    tasks = build_tasks(args.generated_root, source_keys, methods)
    tasks = tasks[args.rank::args.world_size]
    if args.limit:
        tasks = tasks[:args.limit]
    log.info("Rank %d/%d: %d tasks", args.rank, args.world_size, len(tasks))

    source_cache, label_cache, rank_cache = {}, {}, {}
    out_rows = []
    for i, (method, key, variant, synth_path) in enumerate(tasks):
        try:
            if key not in source_cache:
                source_cache[key] = _load(args.images_dir / f"{key}_0000.nii.gz", device)
                label_cache[key]  = _load(args.labels_dir / f"{key}.nii.gz", device).round().long()
                rank_cache[key]   = {r: rank_transform(source_cache[key], r) for r in RANK_RADII}
            source, labels = source_cache[key], label_cache[key]

            if method in CONTROL_METHODS:
                synth = make_control(source, labels, method, variant)
            else:
                synth = _load(synth_path, device)
                if synth.shape != source.shape:
                    log.warning("SHAPE MISMATCH %s %s: %s vs %s — skip",
                                method, key, tuple(synth.shape), tuple(source.shape))
                    continue

            for r in metrics_for_pair(source, synth, labels, args.mi_bins,
                                      args.erode_iters, src_ranks=rank_cache[key]):
                sub = key.split("_ses-")[0]
                ses = "ses-" + key.split("_ses-")[1].replace("_T1w", "")
                out_rows.append(dict(set=args.set_label, method=method,
                                     subject=sub, session=ses, variant=variant, **r))
        except Exception as e:                                        # noqa: BLE001
            log.warning("FAILED %s/%s/%s: %s", method, key, variant, e)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d done", i + 1, len(tasks))

    df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Wrote %d rows → %s", len(df), args.output_csv)


if __name__ == "__main__":
    main()
