#!/usr/bin/env python3
"""
Extract the image assets for the PALETTE-Aug motion video (paper/motion/).

Every transform frame is produced by the REAL transform helpers
(src/synthesis/v26_6_synthesis.py: _kmeans_1d, _voronoi_region_ids), following
paper/scripts/generate_method_figure_panels.py's run_pipeline() (Voronoi split
forced on and every label remapped, so one draw shows every step). The only
addition is a noise-fill variant on the SAME draw (ablation rung 4) next to the
real fill (rung 5), plus the colour maps of the partitions for the animation.

Usage (inside a job, never on a login node):
  python extract_assets.py --repo <repo> --data-root <scratch> --out <remotion>/public/assets
"""
from __future__ import annotations

import argparse
import colorsys
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

SIZE = 768        # hero / texture / contrast stills
TILE = 384        # rapid-fire draw tiles
N_DRAWS = 24

# (task key, display name, dataset root under --data-root, training dataset dir glob,
#  case id, {contrast name: dataset dir glob}, label name for captions)
CASES = [
    ("brain_healthy", "Healthy brain", "on-harmony/2_nnUNet/raw", "Dataset031_*", "sub-03286_ses-NOT2ING001_T1w",
     {"T1w": ("Dataset031_*", "sub-03286_ses-NOT2ING001_T1w"),
      "T2w": ("Dataset032_*", "sub-03286_ses-NOT2ING001_T2w"),
      "DWI": ("Dataset033_*", "sub-03286_ses-NOT2ING001_dwi_ap")}),
    ("brain_tumor", "Brain tumor", "brats2024-glioma/2_nnUNet/raw", "Dataset051_*", "BraTSGLI00005100",
     {"T1n": ("Dataset051_*", "BraTSGLI00005100"), "T1c": ("Dataset054_*", "BraTSGLI00005100"),
      "T2w": ("Dataset052_*", "BraTSGLI00005100"), "FLAIR": ("Dataset053_*", "BraTSGLI00005100")}),
    ("brain_ms", "MS lesions", "open-ms/2_nnUNet_open-ms/raw", "Dataset070_*", "patient01",
     {"FLAIR": ("Dataset070_*", "patient01"), "T1w": ("Dataset071_*", "patient01")}),
    ("abdomen", "Abdominal organs", "chaos/2_nnUNet_chaos/raw", "Dataset061_*", "MR01",
     {"T2 SPIR": ("Dataset061_*", "MR01"), "T1 in-phase": ("Dataset060_*", "MR01")}),
    ("breast", "Breast lesions", "ispy2/2_nnUNet/raw", "Dataset100_*", "ispy2_104384_uni",
     {"T1w CE": ("Dataset100_*", "ispy2_104384_uni"), "T2w": ("Dataset101_*", "ispy2_104384_uni")}),
    ("mandible", "Mandible", "toothfairy2/2_nnUNet/raw", "Dataset110_*", None,
     {"CBCT": ("Dataset110_*", None)}),
    ("pelvis", "Pelvis", "totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw", "Dataset131_*", None,
     {"MRI": ("Dataset131_*", None), "CT": ("Dataset130_*", None)}),
]
HERO = "brain_healthy"
# ToothFairy2 volumes are flipped at source (teeth end up posterior in the axial view); flip for display only
FLIP_AP = {"mandible"}
TEXTURE = ["brain_tumor", "brain_ms"]


# ---------------------------------------------------------------- file lookup
def find_case(root: Path, ds_glob: str, case: str | None):
    dss = sorted(glob.glob(str(root / ds_glob)))
    if not dss:
        raise FileNotFoundError(f"no dataset {ds_glob} under {root}")
    ds = Path(dss[0])
    if case is None:  # first labelled case
        case = sorted(Path(p).name[:-len(".nii.gz")] for p in glob.glob(str(ds / "labelsTr/*.nii.gz")))[0]
    for sub in ("imagesTr", "imagesTs"):
        img = ds / sub / f"{case}_0000.nii.gz"
        if img.exists():
            lbl = ds / sub.replace("images", "labels") / f"{case}.nii.gz"
            return img, (lbl if lbl.exists() else None), case
    raise FileNotFoundError(f"{case} not in {ds}")


def case_ids(root: Path, ds_glob: str, labelled: bool = False) -> set[str]:
    dss = sorted(glob.glob(str(root / ds_glob)))
    if not dss:
        return set()
    pat = "labelsTr/*.nii.gz" if labelled else "images*/*_0000.nii.gz"
    cut = ".nii.gz" if labelled else "_0000.nii.gz"
    return {Path(p).name[:-len(cut)] for p in glob.glob(str(Path(dss[0]) / pat))}


def resolve_cases(root: Path, train_glob: str, case: str | None, contrasts: dict) -> tuple[str | None, dict]:
    """Use the requested case when every contrast has it; otherwise the first labelled
    training case present in every contrast dataset (TamIA holds subsets of some datasets).
    Contrast case ids are mapped by replacing the training case id inside them."""
    if case is None:
        return None, contrasts
    def subst(c, new):
        return c.replace(case, new) if c else c
    def ok(cand):
        return all(subst(cc, cand) in case_ids(root, cg) for cg, cc in contrasts.values())
    if ok(case) and case in case_ids(root, train_glob):
        return case, contrasts
    for cand in sorted(case_ids(root, train_glob, labelled=True)):
        if ok(cand):
            print(f"[info] {case} not available here, using {cand}")
            return cand, {k: (cg, subst(cc, cand)) for k, (cg, cc) in contrasts.items()}
    raise FileNotFoundError(f"no case of {train_glob} present in every contrast under {root}")


# ---------------------------------------------------------------- slicing / orientation
def axial_slice(img_path: Path, lbl_path: Path | None, z_override: float | None = None):
    """Axial slice with the most label voxels, displayed anterior-up and
    radiological (patient right on image left)."""
    img = nib.load(str(img_path))
    codes = nib.aff2axcodes(img.affine)
    ax = next(i for i, c in enumerate(codes) if c in "SI")
    lbl = None
    if lbl_path is not None:
        lbl = np.asanyarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)
        counts = (lbl > 0).sum(axis=tuple(i for i in range(3) if i != ax))
        z = int(np.argmax(counts))
    else:
        z = img.shape[ax] // 2
    if z_override is not None:
        z = int(round(z_override * (img.shape[ax] - 1)))
    sl = [slice(None)] * 3
    sl[ax] = z
    im2 = np.asanyarray(img.dataobj[tuple(sl)]).astype(np.float32)
    lb2 = lbl[tuple(sl)] if lbl is not None else np.zeros(im2.shape, np.int32)
    rest = [codes[i] for i in range(3) if i != ax]
    # want rows = A->P (row 0 anterior), cols = R->L (col 0 patient right)
    if rest[0] in "RL":
        im2, lb2, rest = im2.T, lb2.T, rest[::-1]
    if rest[0] == "P":
        im2, lb2 = im2[::-1], lb2[::-1]
    if rest[1] == "R":
        im2, lb2 = im2[:, ::-1], lb2[:, ::-1]
    return np.ascontiguousarray(im2), np.ascontiguousarray(lb2), z / max(1, img.shape[ax] - 1)


def normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.7)
    return np.clip((a - lo) / (hi - lo + 1e-7), 0, 1).astype(np.float32)


def body_mask(a: np.ndarray) -> np.ndarray:
    """Largest connected component above a fraction of the Otsu threshold, holes filled."""
    from scipy import ndimage as ndi
    hist, edges = np.histogram(a, bins=256, range=(0, 1))
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m = np.cumsum(hist * edges[:-1]); mu0 = m / np.maximum(w0, 1); mu1 = (m[-1] - m) / np.maximum(w1, 1)
    otsu = edges[int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))]
    lab, n = ndi.label(ndi.binary_opening(a > 0.35 * otsu, iterations=1))
    if n == 0:
        return np.ones_like(a, bool)
    keep = lab == (1 + int(np.argmax(ndi.sum(np.ones_like(a), lab, range(1, n + 1)))))
    return ndi.binary_fill_holes(ndi.binary_closing(keep, iterations=3))


def fg_box(a: np.ndarray, pad: int = 6):
    ys, xs = np.where(a > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    # square box so every frame shares one geometry
    h, w = y1 - y0, x1 - x0
    s = max(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - s // 2, cx - s // 2, s


def square_crop(a: np.ndarray, box, fill=0.0):
    y0, x0, s = box
    out = np.full((s, s) + a.shape[2:], fill, dtype=a.dtype)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + s, a.shape[0]), min(x0 + s, a.shape[1])
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


# ---------------------------------------------------------------- saving
def save_gray(a: np.ndarray, path: Path, size: int):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    im.resize((size, size), Image.BICUBIC).save(path, optimize=True)


def save_rgba(rgb: np.ndarray, alpha: np.ndarray, path: Path, size: int):
    arr = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    Image.fromarray(arr, "RGBA").resize((size, size), Image.NEAREST).save(path, optimize=True)


def palette_rgb(ids: np.ndarray, parent: np.ndarray | None = None, n_parent: int = 1):
    """Categorical colours: hue by parent cluster, lightness/sat by child index."""
    hues = [0.60, 0.07, 0.45, 0.88, 0.13, 0.74, 0.33, 0.97, 0.53, 0.20]
    rgb = np.zeros(ids.shape + (3,), np.uint8)
    for v in np.unique(ids):
        if v < 0:
            continue
        if parent is None:
            h = hues[int(v) % len(hues)] if v < len(hues) else (0.618 * v) % 1
            l, s = 0.58, 0.72
        else:
            p = int(np.bincount(parent[ids == v].clip(0)).argmax())
            j = int(v) * 7919 % 7
            h = (hues[p % len(hues)] + (j - 3) * 0.035) % 1
            l, s = 0.34 + 0.06 * j, 0.55 + 0.06 * (j % 4)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        rgb[ids == v] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


# ---------------------------------------------------------------- the transform (real helpers)
def run_draw(img01: np.ndarray, lbl: np.ndarray, seed: int):
    from src.synthesis.v26_6_synthesis import (C_CHOICES, DARK_THRESHOLD, N_KMEANS_SUBSAMPLE,
                                               _kmeans_1d, _voronoi_region_ids)
    torch.manual_seed(seed)
    H, W = img01.shape
    N = H * W
    flat = torch.from_numpy(img01.reshape(-1)).float()
    fgm = (flat > DARK_THRESHOLD).float()
    lbl_flat = torch.from_numpy(lbl.reshape(-1)).long()

    C = C_CHOICES[int(torch.rand(1).item() * len(C_CHOICES))]
    fg_vals = flat[flat > DARK_THRESHOLD]
    fg_vals = fg_vals[torch.randperm(len(fg_vals))[:N_KMEANS_SUBSAMPLE]]
    centroids = _kmeans_1d(fg_vals, C)
    sorted_c, sort_idx = torch.sort(centroids)
    lbl_s = torch.bucketize(flat, (sorted_c[:-1] + sorted_c[1:]) / 2.0)
    lbl_l = sort_idx[lbl_s].long()
    coords = torch.stack(torch.meshgrid(torch.arange(1, dtype=torch.float32),
                                        torch.arange(H, dtype=torch.float32),
                                        torch.arange(W, dtype=torch.float32), indexing="ij"), -1).reshape(N, 3)
    rid, R = _voronoi_region_ids(coords, lbl_l, fgm, C, torch.device("cpu"), force_split=True)

    eps = 1e-7
    n_c = torch.zeros(R).scatter_add_(0, rid, fgm)
    mean_c = torch.zeros(R).scatter_add_(0, rid, flat * fgm) / n_c.clamp(min=eps)
    mu_c = torch.rand(R)
    alpha_c = (torch.rand(R) * 1.5 + 0.5) * ((torch.rand(R) > 0.5).float() * 2 - 1)

    rid_np, cls_np = rid.numpy(), lbl_l.numpy()
    dom = torch.zeros(C)
    for c in range(C):
        regs = np.unique(rid_np[(cls_np == c) & (fgm.numpy() > 0)])
        if len(regs):
            dom[c] = mu_c[regs[int(np.argmax([n_c[r].item() for r in regs]))]]
    p_b = dom[lbl_l] * fgm
    p_c = mu_c[rid] * fgm
    step3 = (mu_c[rid] + alpha_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * fgm
    # rung-4 noise fill on the same regions, as in the paper's fill-swap figure (generate_method_figure_panels.py
    # panel n): mu_c + sigma_c*N(0,1), sigma_c ~ U(0.05, 0.25), own generator so the other draws are unchanged
    g = torch.Generator().manual_seed(1)
    sig_c = torch.rand(R, generator=g) * 0.20 + 0.05
    noise3 = (mu_c[rid] + sig_c[rid] * torch.randn(N, generator=g)).clamp(0, 1) * fgm
    p_d, p_e, p_n = p_c.clone(), step3.clone(), noise3.clone()
    for ell in [int(x) for x in torch.unique(lbl_flat) if x > 0]:
        m = (lbl_flat == ell)
        if m.sum() < 4:
            continue
        mu_l = torch.rand(1).item()
        a_l = (torch.rand(1).item() * 1.5 + 0.5) * (1.0 if torch.rand(1).item() > 0.5 else -1.0)
        # step 4 = Eq. (1) applied again on top of the step-3 image (here its flat preview), so the
        # Voronoi sub-regions inside each label survive (overwriting with mu_l alone erased them)
        p_d = torch.where(m, (mu_l + a_l * (p_c - p_c[m].mean())).clamp(0, 1), p_d)
        p_e = torch.where(m, (mu_l + a_l * (step3 - step3[m].mean())).clamp(0, 1), p_e)
        p_n = torch.where(m, (mu_l + a_l * (noise3 - noise3[m].mean())).clamp(0, 1), p_n)
    rs = lambda t: t.reshape(H, W).numpy()
    return dict(a=img01, b=rs(p_b), c=rs(p_c), d=rs(p_d), e=rs(p_e), noise=rs(p_n),
                cls=np.where(rs(fgm) > 0, cls_np.reshape(H, W), -1),
                rid=np.where(rs(fgm) > 0, rid_np.reshape(H, W), -1),
                C=C, R=int(R), mu=mu_c.numpy(), alpha=alpha_c.numpy(), mean=mean_c.numpy())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True, nargs="+", help="searched in order, per task")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tasks": [], "hero": None, "texture": []}

    for key, name, root, train_glob, case, contrasts in CASES:
        d = out / key
        d.mkdir(exist_ok=True)
        for dr in map(Path, args.data_root):
            try:
                case, contrasts = resolve_cases(dr / root, train_glob, case, contrasts)
                find_case(dr / root, train_glob, case)
                data_root = dr
                break
            except FileNotFoundError as e:
                print(f"[info] {key}: not usable under {dr}: {e}")
        else:
            print(f"[warn] {key}: skipped, no data root has it")
            continue
        img_p, lbl_p, case_id = find_case(data_root / root, train_glob, case)
        im, lb, zrel = axial_slice(img_p, lbl_p)
        im01 = normalize01(im)
        im01 = im01 * body_mask(im01)
        if key in FLIP_AP:
            im01, lb = im01[::-1].copy(), lb[::-1].copy()
        box = fg_box(im01)
        im01c, lbc = square_crop(im01, box), square_crop(lb, box, 0)
        save_gray(im01c, d / "input.png", SIZE)
        lab_rgb = palette_rgb(np.where(lbc > 0, lbc, -1))
        save_rgba(lab_rgb, (lbc > 0).astype(np.float32), d / "labels.png", SIZE)
        entry = {"key": key, "name": name, "case": case_id, "source": img_p.parent.parent.name,
                 "n_labels": int(len(np.unique(lbc)) - 1), "contrasts": [], "draws": N_DRAWS}

        # other contrasts of the same subject (each at the slice with the most label)
        for ci, (cname, (cglob, ccase)) in enumerate(contrasts.items()):
            try:
                cp, clp, _ = find_case(data_root / root, cglob, ccase)
                cim, clb, _ = axial_slice(cp, clp)
                c01 = normalize01(cim)
                if key in FLIP_AP:
                    c01 = c01[::-1].copy()
                c01 = square_crop(c01, fg_box(c01))
                save_gray(c01, d / f"contrast_{ci}.png", SIZE)
                entry["contrasts"].append({"name": cname, "file": f"{key}/contrast_{ci}.png"})
            except Exception as e:  # keep going; the manifest records what exists
                print(f"[warn] {key} {cname}: {e}")

        # rapid-fire training draws
        for s in range(N_DRAWS):
            r = run_draw(im01c, lbc, seed=1000 + s)
            save_gray(r["e"], d / f"draw_{s:02d}.png", TILE)
        print(f"[ok] {key}: {case_id} z={zrel:.2f} labels={entry['n_labels']} contrasts={len(entry['contrasts'])}")

        if key == HERO:
            r = run_draw(im01c, lbc, seed=7)
            for k in "abcde":
                save_gray(r[k], d / f"hero_{k}.png", SIZE)
            fg = (r["cls"] >= 0).astype(np.float32)
            save_rgba(palette_rgb(r["cls"]), fg, d / "hero_cls.png", SIZE)
            save_rgba(palette_rgb(r["rid"], r["cls"], r["C"]), fg, d / "hero_rid.png", SIZE)
            # probe region: the region holding the label-weighted centre of the slice
            ys, xs = np.where(lbc > 0)
            py, px = int(np.median(ys)), int(np.median(xs))
            if r["rid"][py, px] < 0:
                py, px = ys[len(ys) // 2], xs[len(xs) // 2]
            rr = int(r["rid"][py, px])
            S = im01c.shape[0]
            manifest["hero"] = {"key": key, "k": r["C"], "regions": r["R"],
                                "probe": {"x": px / S, "y": py / S, "mu": float(r["mu"][rr]),
                                          "alpha": float(r["alpha"][rr]), "mean": float(r["mean"][rr])}}

        if key in TEXTURE:
            r = run_draw(im01c, lbc, seed=11)
            save_gray(r["noise"], d / "tex_noise.png", SIZE)
            save_gray(r["e"], d / "tex_real.png", SIZE)
            ys, xs = np.where(lbc > 0)
            S = im01c.shape[0]
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            half = max((ys.max() - ys.min()) * 0.8, (xs.max() - xs.min()) * 0.8, S * 0.25)  # zoom <= 2x
            manifest["texture"].append({"key": key, "name": name,
                                        "zoom": {"cx": cx / S, "cy": cy / S, "half": min(0.5, half / S)}})
        manifest["tasks"].append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[done]", out / "manifest.json")


if __name__ == "__main__":
    main()
