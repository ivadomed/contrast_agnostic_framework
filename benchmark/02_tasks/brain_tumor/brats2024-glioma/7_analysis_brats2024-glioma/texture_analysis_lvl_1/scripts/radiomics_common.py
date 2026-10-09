"""Shared PyRadiomics helpers (IBSI-style texture: z-scored intensities, binCount=32, Original+LoG[1,2,3]+Wavelet,
firstorder/glcm/glrlm/glszm/gldm/ngtdm; shape separate, region only). Masks: eroded target region R (1 vox) and ring (1-5 vox)."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
OUT = THIS.parent / "outputs"
DATA, TABLES, PLOTS, LOGS = OUT / "data", OUT / "tables", OUT / "plots", OUT / "logs"
MIN_VOX = 50
PAD_MM = 20


def make_extractors():
    import SimpleITK as sitk
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    kw = dict(binCount=32, normalize=False, preCrop=True, correctMask=True, resampledPixelSpacing=None, label=1)
    tex = featureextractor.RadiomicsFeatureExtractor(**kw)
    tex.disableAllImageTypes()
    tex.enableImageTypeByName("Original")
    tex.enableImageTypeByName("LoG", customArgs={"sigma": [1, 2, 3]})
    tex.enableImageTypeByName("Wavelet")
    tex.disableAllFeatures()
    for c in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        tex.enableFeatureClassByName(c)
    shp = featureextractor.RadiomicsFeatureExtractor(**kw)
    shp.disableAllImageTypes()
    shp.enableImageTypeByName("Original")
    shp.disableAllFeatures()
    shp.enableFeatureClassByName("shape")
    return tex, shp


def _sitk(a, spacing=(1.0, 1.0, 1.0)):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
    im.SetSpacing(spacing)
    return im


def run_ex(ex, z, mask, prefix):
    import SimpleITK as sitk
    img = _sitk(z.astype(np.float32))
    m = _sitk(mask.astype(np.uint8))
    r = ex.execute(img, m, label=1)
    return {f"{prefix}__{k}": float(v) for k, v in r.items() if not k.startswith("diagnostics")}


def masks_from_region(Rc, brc):
    """Rc: region mask (cropped, 1 mm vox), brc: brain/foreground mask. returns eroded R, ring (bigfeat definition)."""
    from scipy.ndimage import distance_transform_edt
    edt_in = distance_transform_edt(Rc)
    Re = edt_in > 1.0
    if int(Re.sum()) < MIN_VOX:
        Re = Rc.copy()
    dout = distance_transform_edt(~Rc)
    ring = (dout > 1.0) & (dout <= 5.0) & brc & ~Rc
    return Re, ring


def region_row(tex, shp, z, Rc, brc):
    """Texture features of R and ring (+ shape of R). Returns dict or None."""
    Re, ring = masks_from_region(Rc, brc)
    if Re.sum() < MIN_VOX or ring.sum() < MIN_VOX:
        return None
    f = {"n_R": int(Re.sum()), "n_ring": int(ring.sum())}
    f.update(run_ex(tex, z, Re, "R"))
    f.update(run_ex(tex, z, ring, "ring"))
    f.update(run_ex(shp, z, Re, "S"))
    return f


def prune_corr(X, thr=0.95):
    """Label-free greedy near-duplicate pruning (|Spearman|>thr). X: rows x feats (NaN allowed). returns kept idx."""
    R = pd.DataFrame(X).rank(axis=0).to_numpy(float)
    R = np.where(np.isnan(R), np.nanmean(R, axis=0, keepdims=True), R)
    sd = R.std(0)
    ok = np.where(sd > 1e-9)[0]
    Z = (R[:, ok] - R[:, ok].mean(0)) / sd[ok]
    C = np.abs(Z.T @ Z) / len(Z)
    kept = []
    for j in range(len(ok)):
        if not kept or C[j, kept].max() < thr:
            kept.append(j)
    return ok[kept]
