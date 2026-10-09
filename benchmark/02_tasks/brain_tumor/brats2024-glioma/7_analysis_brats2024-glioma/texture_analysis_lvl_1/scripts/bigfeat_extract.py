#!/usr/bin/env python
"""
Big image-feature extraction for the fill-swap classifier (bigfeat_*).

Per patient x contrast x region (GT-present): features of R (eroded 1 vox), RING (1-5 vox outside R, inside brain)
and the WHOLE BRAIN, on intensities z-scored within brain (brain = healthy | whole tumor), per contrast.
  first-order (11): mean std p5 p25 p50 p75 p95 iqr skew kurt entropy(32 bins over brain p1-p99)
  texture: hp_std (Gaussian-highpass sigma 2), gm_mean/gm_std (gradient magnitude), lap_std, acf 12 (axis x lag 1-4,
           on highpass), GLCM (contrast homogeneity energy correlation dissimilarity; 32 levels, 4 angles, axial slices),
           LBP(8,1,uniform) histogram entropy
  contrast (R vs ring): d (mean diff / pooled std), auc (max(AUC,1-AUC)), ks
  boundary: border-shell mean gradient magnitude / brain mean gradient magnitude
  shape: log10 volume, surface/volume, centroid distance to brain centroid (vox), fraction of ring that is other tumor
Sharded + resumable: --rank R --world-size W -> outputs/data/bigfeat_shard{R}.csv (one append per patient).
Brain-level features are region independent (same value on every region row of a patient x contrast).
  python bigfeat_extract.py --rank 0 --world-size 4 [--limit-patients N] [--shard-prefix P]
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace
from scipy.stats import kurtosis, ks_2samp, rankdata, skew
from skimage.feature import graycomatrix, local_binary_pattern

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from compute_cross_contrast_ngf import CONTRAST_SUFFIX, LABEL_IDS, load_patient, region_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DATA = THIS.parent / "outputs" / "data"
REGIONS = ("SNFH", "RC", "ET", "NCR")
LAGS = (1, 2, 3, 4)
MARGIN = 7
RING_D = 5.0
MIN_VOX = 20
MAX_SLICES = 8
SLICE_AXIS = 2
RNG = np.random.default_rng(0)
GL_NAMES = ("glcm_contrast", "glcm_homog", "glcm_energy", "glcm_corr", "glcm_dissim", "lbp_ent")


def first_order(v, lo, hi):
    f = {}
    if v.size < MIN_VOX:
        return {k: np.nan for k in ("mean", "std", "p5", "p25", "p50", "p75", "p95", "iqr", "skew", "kurt", "ent")}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    f.update(mean=v.mean(), std=v.std(), p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4], iqr=q[3] - q[1],
             skew=float(skew(v)), kurt=float(kurtosis(v)))
    h, _ = np.histogram(np.clip(v, lo, hi), bins=32, range=(lo, hi))
    p = h[h > 0] / h.sum()
    f["ent"] = float(-(p * np.log2(p)).sum())
    return f


def acf_vec(img, mask):
    out = {}
    for ax in range(3):
        for k in LAGS:
            key = f"acf_ax{ax}_lag{k}"
            n = mask.shape[ax]
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, n - k), slice(k, n)
            both = mask[tuple(lo)] & mask[tuple(hi)]
            val = np.nan
            if both.sum() >= 100:
                x, y = img[tuple(lo)][both], img[tuple(hi)][both]
                sx, sy = x.std(), y.std()
                if sx > 0 and sy > 0:
                    val = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
            out[key] = val
    return out


def glcm_lbp(z, mask, lo, hi):
    """GLCM props (avg over 4 angles & slices) + LBP-hist entropy on axial slices through mask."""
    names = ("glcm_contrast", "glcm_homog", "glcm_energy", "glcm_corr", "glcm_dissim", "lbp_ent")
    out = {k: np.nan for k in names}
    sl = np.where(mask.sum(axis=tuple(a for a in range(3) if a != SLICE_AXIS)) >= 30)[0]
    if len(sl) == 0:
        return out
    if len(sl) > MAX_SLICES:
        sl = sl[np.linspace(0, len(sl) - 1, MAX_SLICES).round().astype(int)]
    idx = np.arange(33)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    acc = []
    lbp_hist = np.zeros(10)
    for s in sl:
        m2 = np.take(mask, s, axis=SLICE_AXIS)
        z2 = np.take(z, s, axis=SLICE_AXIS)
        xs, ys = np.where(m2)
        x0, x1, y0, y1 = max(xs.min() - 1, 0), xs.max() + 2, max(ys.min() - 1, 0), ys.max() + 2
        m2, z2 = m2[x0:x1, y0:y1], z2[x0:x1, y0:y1]
        q = np.where(m2, np.clip((z2 - lo) / (hi - lo) * 31, 0, 31).astype(np.uint8) + 1, 0).astype(np.uint8)
        P = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True, normed=False)[:, :, 0, :]
        P[0, :, :] = 0
        P[:, 0, :] = 0
        vals = []
        for a in range(P.shape[2]):
            Pa = P[:, :, a].astype(float)
            tot = Pa.sum()
            if tot < 20:
                continue
            Pa /= tot
            mi, mj = (Pa * I).sum(), (Pa * J).sum()
            si, sj = np.sqrt((Pa * (I - mi) ** 2).sum()), np.sqrt((Pa * (J - mj) ** 2).sum())
            corr = ((Pa * (I - mi) * (J - mj)).sum() / (si * sj)) if si > 0 and sj > 0 else np.nan
            vals.append([(Pa * (I - J) ** 2).sum(), (Pa / (1 + (I - J) ** 2)).sum(), np.sqrt((Pa ** 2).sum()), corr,
                         (Pa * np.abs(I - J)).sum()])
        if vals:
            acc.append(np.nanmean(np.array(vals), axis=0))
        lbp = local_binary_pattern(z2.astype(np.float64), 8, 1, method="uniform").astype(int)
        lbp_hist += np.bincount(lbp[m2], minlength=10)[:10]
    if acc:
        a = np.nanmean(np.array(acc), axis=0)
        out.update(glcm_contrast=a[0], glcm_homog=a[1], glcm_energy=a[2], glcm_corr=a[3], glcm_dissim=a[4])
    if lbp_hist.sum() >= MIN_VOX:
        p = lbp_hist[lbp_hist > 0] / lbp_hist.sum()
        out["lbp_ent"] = float(-(p * np.log2(p)).sum())
    return out


def part_feats(z, hp, gm, lap, mask, lo, hi):
    f = first_order(z[mask], lo, hi)
    if mask.sum() >= MIN_VOX:
        f["hp_std"] = float(hp[mask].std())
        f["gm_mean"] = float(gm[mask].mean())
        f["gm_std"] = float(gm[mask].std())
        f["lap_std"] = float(lap[mask].std())
    else:
        f.update(hp_std=np.nan, gm_mean=np.nan, gm_std=np.nan, lap_std=np.nan)
    f.update(acf_vec(hp, mask) if mask.sum() >= MIN_VOX else {f"acf_ax{a}_lag{k}": np.nan for a in range(3) for k in LAGS})
    f.update(glcm_lbp(z, mask, lo, hi) if mask.sum() >= MIN_VOX else {k: np.nan for k in GL_NAMES})
    return f


def sub(a, n, rng=RNG):
    return a if a.size <= n else rng.choice(a, n, replace=False)


def patient_rows(pid):
    vols, label = load_patient(pid, "cpu")
    masks = {k: m.numpy() for k, m in region_masks(label, vols["t1n"]).items()}
    lab = label.numpy()
    brain = masks["healthy"] | masks["whole_tumor"]
    bidx = np.argwhere(brain)
    bcent = bidx.mean(0)
    # per contrast whole-volume maps
    maps = {}
    for c in CONTRAST_SUFFIX:
        x = vols[c].numpy().astype(np.float32)
        v = x[brain]
        z = np.where(brain, (x - v.mean()) / max(v.std(), 1e-6), 0).astype(np.float32)
        b = brain.astype(np.float32)
        smooth = gaussian_filter(z * b, 2.0) / np.maximum(gaussian_filter(b, 2.0), 1e-6)
        hp = np.where(brain, z - smooth, 0).astype(np.float32)
        g = np.gradient(z)
        gm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2).astype(np.float32)
        lap = laplace(z).astype(np.float32)
        lo, hi = np.percentile(z[brain], [1, 99])
        maps[c] = dict(z=z, hp=hp, gm=gm, lap=lap, lo=float(lo), hi=float(hi), gm_brain=float(gm[brain].mean()))
    brain_f = {}
    for c, M in maps.items():
        brain_f[c] = {f"brain_{k}": v for k, v in part_feats(M["z"], M["hp"], M["gm"], M["lap"], brain, M["lo"], M["hi"]).items()}
    rows = []
    for region in REGIONS:
        R = masks[region]
        n_full = int(R.sum())
        if n_full == 0:
            continue
        idx = np.argwhere(R)
        lo_i = np.maximum(idx.min(0) - MARGIN, 0)
        hi_i = np.minimum(idx.max(0) + MARGIN + 1, R.shape)
        cs = tuple(slice(a, b) for a, b in zip(lo_i, hi_i))
        Rc, brc, labc = R[cs], brain[cs], lab[cs]
        edt_in = distance_transform_edt(Rc)
        Re = edt_in > 1.0
        if int(Re.sum()) < MIN_VOX:
            Re = Rc.copy()
        shell = Rc & ~(edt_in > 1.0)
        dout = distance_transform_edt(~Rc)
        ring = (dout > 1.0) & (dout <= RING_D) & brc & ~Rc
        other_tumor = float(((labc > 0) & ~Rc & ring).sum() / max(ring.sum(), 1))
        cen = np.argwhere(Rc).mean(0) + lo_i
        shape = dict(shape_logvol=float(np.log10(n_full)), shape_surf_vol=float(shell.sum() / n_full),
                     shape_cent_dist=float(np.linalg.norm(cen - bcent)), shape_ring_other_tumor=other_tumor)
        for c, M in maps.items():
            zc, hpc, gmc, lapc = M["z"][cs], M["hp"][cs], M["gm"][cs], M["lap"][cs]
            f = {"patient": pid, "contrast": c, "region": region, "n_R": int(Re.sum()), "n_ring": int(ring.sum())}
            f.update({f"R_{k}": v for k, v in part_feats(zc, hpc, gmc, lapc, Re, M["lo"], M["hi"]).items()})
            f.update({f"ring_{k}": v for k, v in part_feats(zc, hpc, gmc, lapc, ring, M["lo"], M["hi"]).items()})
            f.update(brain_f[c])
            a, b = zc[Re], zc[ring]
            if a.size >= MIN_VOX and b.size >= MIN_VOX:
                pooled = np.sqrt((a.var() + b.var()) / 2)
                f["con_d"] = float((a.mean() - b.mean()) / max(pooled, 1e-6))
                a2, b2 = sub(a, 20000), sub(b, 20000)
                r = rankdata(np.concatenate([a2, b2]))
                auc = (r[: len(a2)].sum() - len(a2) * (len(a2) + 1) / 2) / (len(a2) * len(b2))
                f["con_auc"] = float(max(auc, 1 - auc))
                f["con_ks"] = float(ks_2samp(a2, b2).statistic)
            else:
                f.update(con_d=np.nan, con_auc=np.nan, con_ks=np.nan)
            f["bnd_grad_ratio"] = float(gmc[shell].mean() / M["gm_brain"]) if shell.sum() >= MIN_VOX else np.nan
            f.update(shape)
            rows.append(f)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--shard-prefix", default="bigfeat_shard")
    a = p.parse_args()
    pts = sorted(pd.read_csv(DATA / "patient_region_deltas.csv")["case"].unique())
    if a.limit_patients:
        pts = pts[: a.limit_patients]
    mine = pts[a.rank::a.world_size]
    shard = DATA / f"{a.shard_prefix}{a.rank}.csv"
    done = set(pd.read_csv(shard)["patient"].unique()) if shard.exists() else set()
    todo = [q for q in mine if q not in done]
    log.info("rank %d/%d: %d assigned, %d done, %d todo", a.rank, a.world_size, len(mine), len(done), len(todo))
    header = not shard.exists()
    for i, pid in enumerate(todo):
        try:
            rows = patient_rows(pid)
        except (FileNotFoundError, ValueError) as e:
            log.warning("SKIP %s: %s", pid, e)
            continue
        pd.DataFrame(rows).to_csv(shard, mode="a", header=header, index=False)
        header = False
        log.info("rank %d: %d/%d (%s, %d rows)", a.rank, i + 1, len(todo), pid, len(rows))
    log.info("rank %d finished", a.rank)


if __name__ == "__main__":
    main()
