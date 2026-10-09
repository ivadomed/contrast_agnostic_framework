#!/usr/bin/env python
"""
Texture COMPLEXITY of the surroundings (ring 1-5 vox outside the region, inside brain) and of the region itself (eroded
1 vox), per patient x contrast x region, BraTS2024-glioma (Paul's idea, 2026-10-07). Same loading / z-scoring / masks as
bigfeat_extract.py (imported), new multi-scale measures that the 311-feature set did not carry:
  dog_slope    slope of log(band-pass energy) vs log(scale) over DoG bands s=(0.5,1,2,4) [G(s)-G(2s) of z, masked
               normalisation]; more negative = energy concentrated at the finest scale (fine, noisy texture),
               near 0 = scale-free / structured texture.
  ms_ratio     log(E_0.5 / E_4): fine-vs-coarse energy ratio.
  vario_slope  slope of log(semivariogram) vs log(lag), lags 1..4 vox (pairs both inside the mask, 3 axes averaged);
               = 2H; fractal dimension fd = 3 - slope/2 (higher = rougher / more complex).
  lstd_cv, lstd_ent  spatial variability of local std (sigma=1 window): coefficient of variation and 32-bin entropy
               over the mask (heterogeneity of the texture itself).
Writes outputs/data/ring_complexity_shard{rank}.csv (one append per patient; resumable). CPU job only.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from bigfeat_extract import DATA, MARGIN, MIN_VOX, REGIONS, RING_D  # noqa: E402
from compute_cross_contrast_ngf import CONTRAST_SUFFIX, load_patient, region_masks  # noqa: E402

SCALES = (0.5, 1.0, 2.0, 4.0)
LAGS = (1, 2, 3, 4)


def masked_gauss(z, b, s):
    return gaussian_filter(z * b, s) / np.maximum(gaussian_filter(b, s), 1e-6)


def complexity(bands, lstd, z, mask):
    f = {}
    if int(mask.sum()) < MIN_VOX:
        return {k: np.nan for k in ("dog_slope", "ms_ratio", "vario_slope", "fd", "lstd_cv", "lstd_ent")}
    e = np.array([bands[s][mask].std() for s in SCALES]) + 1e-8
    f["dog_slope"] = float(np.polyfit(np.log(SCALES), np.log(e), 1)[0])
    f["ms_ratio"] = float(np.log(e[0] / e[-1]))
    g = []
    for h in LAGS:
        acc, n = 0.0, 0
        for ax in range(3):
            a = np.take(z, np.arange(z.shape[ax] - h), axis=ax); b = np.take(z, np.arange(h, z.shape[ax]), axis=ax)
            ma = np.take(mask, np.arange(mask.shape[ax] - h), axis=ax) & np.take(mask, np.arange(h, mask.shape[ax]), axis=ax)
            if ma.sum() >= 50:
                acc += float(((a - b)[ma] ** 2).mean()); n += 1
        g.append(acc / n if n else np.nan)
    g = np.array(g)
    if np.isfinite(g).sum() >= 3 and np.all(g[np.isfinite(g)] > 0):
        ok = np.isfinite(g); sl = float(np.polyfit(np.log(np.array(LAGS)[ok]), np.log(g[ok]), 1)[0])
        f["vario_slope"] = sl; f["fd"] = 3.0 - sl / 2.0
    else:
        f["vario_slope"] = f["fd"] = np.nan
    v = lstd[mask]
    f["lstd_cv"] = float(v.std() / max(v.mean(), 1e-6))
    hist, _ = np.histogram(v, bins=32, range=(0, max(float(np.percentile(v, 99)), 1e-6)))
    p = hist / max(hist.sum(), 1); p = p[p > 0]
    f["lstd_ent"] = float(-(p * np.log(p)).sum())
    return f


def patient_rows(pid):
    vols, label = load_patient(pid, "cpu")
    masks = {k: m.numpy() for k, m in region_masks(label, vols["t1n"]).items()}
    brain = masks["healthy"] | masks["whole_tumor"]
    b = brain.astype(np.float32)
    maps = {}
    for c in CONTRAST_SUFFIX:
        x = vols[c].numpy().astype(np.float32); v = x[brain]
        z = np.where(brain, (x - v.mean()) / max(v.std(), 1e-6), 0).astype(np.float32)
        bands = {s: (masked_gauss(z, b, s) - masked_gauss(z, b, 2 * s)).astype(np.float32) for s in SCALES}
        m1, m2 = masked_gauss(z, b, 1.0), masked_gauss(z * z, b, 1.0)
        lstd = np.sqrt(np.maximum(m2 - m1 ** 2, 0)).astype(np.float32)
        maps[c] = (z, bands, lstd)
    rows = []
    for region in REGIONS:
        R = masks[region]
        if R.sum() == 0:
            continue
        idx = np.argwhere(R); lo = np.maximum(idx.min(0) - MARGIN, 0); hi = np.minimum(idx.max(0) + MARGIN + 1, R.shape)
        cs = tuple(slice(a, b_) for a, b_ in zip(lo, hi))
        Rc, brc = R[cs], brain[cs]
        Re = distance_transform_edt(Rc) > 1.0
        if int(Re.sum()) < MIN_VOX:
            Re = Rc.copy()
        dout = distance_transform_edt(~Rc)
        ring = (dout > 1.0) & (dout <= RING_D) & brc & ~Rc
        for c, (z, bands, lstd) in maps.items():
            bc = {s: bands[s][cs] for s in SCALES}
            f = dict(patient=pid, contrast=c, region=region, n_R=int(Re.sum()), n_ring=int(ring.sum()))
            f.update({f"ring_{k}": v for k, v in complexity(bc, lstd[cs], z[cs], ring).items()})
            f.update({f"R_{k}": v for k, v in complexity(bc, lstd[cs], z[cs], Re).items()})
            rows.append(f)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0); p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit-patients", type=int, default=None)
    a = p.parse_args()
    pats = sorted(pd.read_csv(DATA / "patient_region_deltas.csv")["case"].unique())[a.rank::a.world_size]
    if a.limit_patients:
        pats = pats[: a.limit_patients]
    shard = DATA / f"ring_complexity_shard{a.rank}.csv"
    done = set(pd.read_csv(shard)["patient"]) if shard.exists() else set()
    for i, pid in enumerate(pats):
        if pid in done:
            continue
        rows = patient_rows(pid)
        pd.DataFrame(rows).to_csv(shard, mode="a", header=not shard.exists(), index=False)
        print(f"{i + 1}/{len(pats)} {pid} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
