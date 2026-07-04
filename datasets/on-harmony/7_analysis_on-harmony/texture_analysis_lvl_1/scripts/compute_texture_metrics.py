#!/usr/bin/env python
"""
Level-1 texture / structure-preservation metrics (input-space, network-free).

For each generated volume (PALETTE / SynthSeg-EM / SynthSeg-noEM / auglab_default) and
each of the 31 anatomical ROIs, measure how much of the SOURCE T1w's structure/texture
survives in the augmented volume, using two directly-citable, contrast-AND-inversion-
invariant metrics (see ../LITERATURE_REVIEW.md):

  * NMI   — Studholme normalized mutual information  = (H(X)+H(Y))/H(X,Y) ∈ [1,2].
            2 = identical dependence, 1 = independent. (Maes 1997 / Studholme 1999)
  * |LNCC|— |local normalized cross-correlation|, box window. (Avants 2008 / ANTs)
            Invariant to local linear intensity change; |·| handles PALETTE's α<0 inversion.

Two INLINE positive controls are also scored directly from the source (no pre-generated
volumes needed): `gamma` and `histeq` — image-driven ops that SHOULD preserve texture,
proving the metric rewards any texture-preservation, not something tuned to PALETTE.

Metrics are computed on ERODED ROI masks (drop boundary voxels → removes PALETTE's
Voronoi boundary-artifact confound). Output is one long-format CSV.

Volumes must be voxel-aligned to their source (guaranteed by the generation step).

Usage
-----
  # self-test the metric math (no data needed; run this first):
  python compute_texture_metrics.py --sanity

  # full run, one GPU, sharded (launch 4 of these, rank 0..3, via set_slot 0..3):
  python compute_texture_metrics.py --device cuda --rank 0 --world-size 4 \
      --output-csv .../outputs/data/metrics_rank0.csv
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
# repo root = .../mri_synthesis_project (parents: scripts→lvl1→7_analysis→on-harmony→datasets→root)
PROJECT_ROOT  = THIS_DIR.parents[5]

DEFAULT_GENERATED = ANALYSIS_ROOT / "data" / "generated"
DEFAULT_DS        = PROJECT_ROOT / "datasets" / "on-harmony" / "2_nnUNet_on-harmony" / "raw" / "Dataset031_OnHarmonyT1w31"
DEFAULT_IMAGES    = DEFAULT_DS / "imagesTr"
DEFAULT_LABELS    = DEFAULT_DS / "labelsTr"
DEFAULT_OUT       = ANALYSIS_ROOT / "outputs" / "data" / "texture_metrics.csv"

GENERATED_METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
CONTROL_METHODS   = ["gamma", "histeq"]          # synthesised inline from source
GAMMA_VALUES      = [0.5, 1.5, 2.5]

MIN_VOX   = 50        # ROI smaller than this (after erosion) → skip
EPS       = 1e-7


# ─────────────────────────── metric core (torch) ────────────────────────────
def _box_filter(x: torch.Tensor, win: int) -> torch.Tensor:
    """Mean over a win^3 box, stride 1, edge-safe. x: (1,1,D,H,W)."""
    return F.avg_pool3d(x, kernel_size=win, stride=1, padding=win // 2,
                        count_include_pad=False)


def lncc_map(x: torch.Tensor, y: torch.Tensor, win: int) -> torch.Tensor:
    """Per-voxel local normalized cross-correlation. x,y: (1,1,D,H,W) → (D,H,W)."""
    mu_x, mu_y = _box_filter(x, win), _box_filter(y, win)
    var_x = _box_filter(x * x, win) - mu_x * mu_x
    var_y = _box_filter(y * y, win) - mu_y * mu_y
    cov   = _box_filter(x * y, win) - mu_x * mu_y
    cc    = cov / (torch.sqrt(var_x.clamp_min(0) * var_y.clamp_min(0)) + EPS)
    return cc[0, 0].clamp(-1.0, 1.0)


def nmi_1d(xv: torch.Tensor, yv: torch.Tensor, bins: int) -> float:
    """Studholme NMI = (H(X)+H(Y))/H(X,Y) on ROI voxel values xv, yv (1-D)."""
    def quant(v):
        vmin, vmax = v.min(), v.max()
        if (vmax - vmin) < EPS:
            return None
        return ((v - vmin) / (vmax - vmin) * (bins - 1)).round().long().clamp(0, bins - 1)

    xi, yi = quant(xv), quant(yv)
    if xi is None or yi is None:               # constant region → undefined
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
def metrics_for_pair(source: torch.Tensor, synth: torch.Tensor, labels: torch.Tensor,
                     bins: int, win: int, erode_iters: int) -> list[dict]:
    """source/synth/labels: (D,H,W) on device. Returns one row-dict per ROI."""
    cc = lncc_map(source[None, None], synth[None, None], win)          # (D,H,W)
    rows = []
    for roi in range(1, 32):                                           # 31 classes
        m = erode(labels == roi, erode_iters)
        n = int(m.sum().item())
        if n < MIN_VOX:
            continue
        rows.append(dict(
            roi_id=roi, n_vox=n,
            nmi=nmi_1d(source[m], synth[m], bins),
            lncc=float(cc[m].abs().mean().item()),
        ))
    return rows


# ─────────────────────────── I/O helpers ────────────────────────────────────
def _load(path: Path, device, order: int) -> torch.Tensor:
    import nibabel as nib
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def _foreground_norm01(x: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    vals = x[fg]
    lo, hi = torch.quantile(vals, 0.01), torch.quantile(vals, 0.99)
    return ((x - lo) / (hi - lo + EPS)).clamp(0, 1)


def make_control(source: torch.Tensor, labels: torch.Tensor, kind: str, param) -> torch.Tensor:
    """Inline positive control synthesised from the source."""
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
def list_source_keys(images_dir: Path) -> dict[str, Path]:
    keys = {}
    for p in sorted(images_dir.glob("*_0000.nii.gz")):
        keys[p.name[:-len("_0000.nii.gz")]] = p       # sub-XXXX_ses-YYYY_T1w
    return keys


def build_tasks(generated_root: Path, source_keys: dict, methods: list[str]) -> list[tuple]:
    """Each task = (method, key, variant, synth_path|None)."""
    tasks = []
    for method in methods:
        if method in CONTROL_METHODS:
            for key in source_keys:
                variants = GAMMA_VALUES if method == "gamma" else [0]
                for v in variants:
                    tasks.append((method, key, v, None))
            continue
        mdir = generated_root / method
        for key in source_keys:
            vol_dir = mdir / key
            for synth in sorted(vol_dir.glob(f"{key}_run-*.nii.gz")):
                variant = int(synth.stem.split("run-")[-1].split(".")[0])
                tasks.append((method, key, variant, synth))
    return tasks


# ─────────────────────────── sanity self-test ───────────────────────────────
def run_sanity(device, bins, win, erode_iters) -> int:
    torch.manual_seed(0)
    D = 40
    # smooth random source (spatially correlated → has texture)
    base = torch.randn(1, 1, D, D, D, device=device)
    source = _box_filter(base, 5)[0, 0]
    source = (source - source.min()) / (source.max() - source.min() + EPS)
    labels = torch.zeros(D, D, D, dtype=torch.long, device=device)
    labels[5:35, 5:35, 5:20] = 1
    labels[5:35, 5:35, 20:35] = 2

    cases = {
        "identity": source.clone(),
        "gamma":    source.clamp_min(0) ** 2.0,
        "noise":    torch.rand(D, D, D, device=device),      # independent noise
    }
    print(f"{'case':10s} {'NMI(mean)':>10s} {'|LNCC|(mean)':>12s}")
    results = {}
    for name, synth in cases.items():
        rows = metrics_for_pair(source, synth, labels, bins, win, erode_iters)
        nmi = float(np.nanmean([r["nmi"] for r in rows]))
        lncc = float(np.nanmean([r["lncc"] for r in rows]))
        results[name] = (nmi, lncc)
        print(f"{name:10s} {nmi:10.3f} {lncc:12.3f}")

    ok = True
    checks = [
        ("identity NMI≈2",    results["identity"][0] > 1.90),
        ("identity |LNCC|≈1", results["identity"][1] > 0.95),
        ("gamma NMI high",    results["gamma"][0]    > 1.50),
        ("gamma |LNCC| high", results["gamma"][1]    > 0.70),
        ("noise NMI≈1",       results["noise"][0]    < 1.15),
        ("noise |LNCC|≈0",    results["noise"][1]    < 0.15),
    ]
    print()
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ─────────────────────────── main ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--methods", type=str, default=",".join(GENERATED_METHODS + CONTROL_METHODS))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mi-bins", type=int, default=64)
    p.add_argument("--lncc-window", type=int, default=9)
    p.add_argument("--erode-iters", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="debug: cap #tasks")
    p.add_argument("--sanity", action="store_true", help="run metric self-test and exit")
    args = p.parse_args()

    device = torch.device(args.device)

    if args.sanity:
        sys.exit(run_sanity(device, args.mi_bins, args.lncc_window, args.erode_iters))

    import pandas as pd
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    source_keys = list_source_keys(args.images_dir)
    if not source_keys:
        log.error("No source images found in %s", args.images_dir); sys.exit(1)
    log.info("Found %d source volumes; methods=%s", len(source_keys), methods)

    tasks = build_tasks(args.generated_root, source_keys, methods)
    tasks = tasks[args.rank::args.world_size]
    if args.limit:
        tasks = tasks[:args.limit]
    log.info("Rank %d/%d: %d tasks", args.rank, args.world_size, len(tasks))

    label_cache: dict[str, torch.Tensor] = {}
    source_cache: dict[str, torch.Tensor] = {}
    out_rows = []
    for i, (method, key, variant, synth_path) in enumerate(tasks):
        try:
            if key not in source_cache:
                source_cache[key] = _load(args.images_dir / f"{key}_0000.nii.gz", device, 1)
                label_cache[key]  = _load(args.labels_dir / f"{key}.nii.gz", device, 0).round().long()
            source, labels = source_cache[key], label_cache[key]

            if method in CONTROL_METHODS:
                synth = make_control(source, labels, method, variant)
            else:
                synth = _load(synth_path, device, 1)
                if synth.shape != source.shape:
                    log.warning("SHAPE MISMATCH %s vs source %s — skipping %s/%s/%s",
                                tuple(synth.shape), tuple(source.shape), method, key, variant)
                    continue

            for r in metrics_for_pair(source, synth, labels, args.mi_bins,
                                      args.lncc_window, args.erode_iters):
                sub, ses = key.split("_ses-")[0], "ses-" + key.split("_ses-")[1].replace("_T1w", "")
                out_rows.append(dict(method=method, subject=sub, session=ses,
                                     variant=variant, **r))
        except Exception as e:                                        # noqa: BLE001
            log.warning("FAILED %s/%s/%s: %s", method, key, variant, e)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d tasks done", i + 1, len(tasks))

    df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Wrote %d rows → %s", len(df), args.output_csv)


if __name__ == "__main__":
    main()
