#!/usr/bin/env python
"""
CONTROL A1: is the CHAOS "no regional texture" result an upsampling artifact?

CHAOS is 2.9-6.6x anisotropic (5.5-9.0mm slices) and is resampled to 1mm isotropic before any cue
is computed, which INVENTS through-plane detail by interpolation. If that interpolation is what
flattens the ROI-vs-shell HOG contrast, the CHAOS null is a processing artifact and the headline
cross-dataset texture comparison collapses.

Test: take open-ms, which is natively 1mm isotropic and shows a positive regional HOG signal.
Impose CHAOS-like anisotropy on it (decimate z by FACTOR, then interpolate back to 1mm, exactly the
round trip CHAOS undergoes) and re-measure the same statistic on the same lesions.

  signal survives  -> interpolation does not destroy the measure -> the CHAOS null is anatomy
  signal collapses -> the CHAOS null is an artifact of its native anisotropy

Usage: python control_anisotropy.py --images DIR --labels DIR --n 12 --factor 6
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import nibabel as nib, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_metrics import regional_texture_contrast
from compute_label_cues import foreground

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def anisotropise(vol: torch.Tensor, factor: int, mode: str) -> torch.Tensor:
    """Decimate along z by `factor` then interpolate back -- the exact round trip CHAOS undergoes."""
    Z = vol.shape[0]
    small = max(2, int(round(Z / factor)))
    kw = {} if mode == "nearest" else {"align_corners": False}
    down = F.interpolate(vol[None, None], size=(small, vol.shape[1], vol.shape[2]), mode=mode, **kw)
    return F.interpolate(down, size=tuple(vol.shape), mode=mode, **kw)[0, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--factor", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(0)
    cases = sorted(p.name[:-len("_0000.nii.gz")] for p in a.images.glob("*_0000.nii.gz"))[:a.n]

    out = {"native": [], "anisotropised": []}
    for c in cases:
        vol = torch.from_numpy(np.asanyarray(nib.load(str(a.images / f"{c}_0000.nii.gz")).dataobj)
                               .astype(np.float32)).to(dev)
        lab = torch.from_numpy(np.asanyarray(nib.load(str(a.labels / f"{c}.nii.gz")).dataobj)
                               .astype(np.float32)).to(dev).round().bool()
        fg = foreground(vol)
        v = (vol - vol[fg].mean()) / vol[fg].std().clamp_min(1e-6)
        for tag, vv, ll in (("native", v, lab),
                            ("anisotropised", anisotropise(v, a.factor, "trilinear"),
                             anisotropise(lab.float(), a.factor, "nearest").bool())):
            r = regional_texture_contrast(vv, ll, fg, gen=gen)
            z = r.get("hog_s0_z", float("nan"))
            if not np.isnan(z):
                out[tag].append(z)
            log.info("%s %-14s hog_s0_z=%+.2f", c, tag, z)
    for k, v in out.items():
        if v:
            log.info("RESULT %-14s mean z=%+.2f  median=%+.2f  frac(z>0)=%.2f  n=%d",
                     k, float(np.mean(v)), float(np.median(v)), float((np.array(v) > 0).mean()), len(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
