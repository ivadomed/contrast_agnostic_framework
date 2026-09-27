#!/usr/bin/env python
"""
Directional cross-contrast "containment" per tumor region: correlation ratio eta^2(A|B).

Question (the t1n<->t2w asymmetry): is the training contrast's intensity structure inside a
region RECOVERABLE from the eval contrast? If T1n's pattern is a function of T2w's but not
vice versa, a T1n-trained model finds what it learned in T2w (success), while a T2w-trained
model looks for structure T1n doesn't carry (failure). NGF (symmetric) cannot express that.

  eta^2(A|B) = Var(E[A|B]) / Var(A)      (Roche et al., MICCAI 1998 — multimodal registration)

= fraction of A's within-region variance explained by ANY function of B; asymmetric by
construction (Roche et al. state this explicitly; LNCS 1496:1115-1124, doi:10.1007/BFb0056301;
also FSL FLIRT's default cost). Estimated by binning B into K equal-frequency bins inside the
region (ties kept in one bin). Invariances: Roche et al. claim only invariance to multiplicative
rescaling of A; offset invariance of A follows trivially (variance ratio), and invariance to any
monotone remap of B is a property of OUR quantile binning (bin membership unchanged) — derived
here, not claimed in the paper. Together these cover the per-region affine remaps (incl. sign
flips) that real-fill applies. Also reported:
  eps2  = Kelley's bias-corrected version (ANOVA effect size): eta^2 has a positive small-sample
          bias ~ (K-1)/N, identical in both directions, eps2 removes it.
  (Theil's uncertainty coefficient I(A;B)/H(A) was considered and DROPPED: with both images
  quantile-binned, H(A) = H(B) = log K, so U(A|B) = U(B|A) — it loses its asymmetry, verified
  by the self-test. eta^2 stays asymmetric because A enters with its actual values.)
Variants: "raw" intensities, and "highpass" = image - Gaussian(sigma=2 vox) so slow shading /
bias field inside a region cannot dominate; K in {16, 32, 64} for robustness. A (region, K)
combination is skipped when the region has fewer than 10*K voxels.

Parallel/resumable exactly like compute_cross_contrast_ngf.py (per-rank shard CSV appended per
patient; rerun skips done patients). Patients = those in outputs/data/patient_region_deltas.csv.

  python compute_correlation_ratio.py --sanity
  python compute_correlation_ratio.py --rank R --world-size W
  (merge: plain concatenation of shard CSVs, done by the summary script)
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import CONTRAST_SUFFIX, load_patient, region_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = THIS_DIR.parent / "outputs" / "data"
KS = (16, 32, 64)
HIGHPASS_SIGMA = 2.0
MIN_VOX_PER_BIN = 10
COLS = ["patient", "region", "variant", "K", "pred", "cond", "n_vox", "eta2", "eps2"]


def quantile_bins(x: np.ndarray, K: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, K + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def eta2_eps2(a: np.ndarray, b_bins: np.ndarray, K: int):
    n = len(a)
    ac = a - a.mean()
    sst = float((ac * ac).sum())
    if sst <= 0:
        return np.nan, np.nan
    cnt = np.bincount(b_bins, minlength=K)
    s = np.bincount(b_bins, weights=ac, minlength=K)
    nz = cnt > 0
    ssb = float((s[nz] ** 2 / cnt[nz]).sum())
    k_eff = int(nz.sum())
    eta2 = ssb / sst
    eps2 = 1.0 - ((sst - ssb) / (n - k_eff)) / (sst / (n - 1)) if n > k_eff else np.nan
    return eta2, eps2


def run_sanity() -> int:
    rng = np.random.default_rng(0)
    n, K = 20000, 32
    # bounded support: the binned estimator is exact up to within-bin variation, which a
    # heavy-tailed B (e.g. exp(normal)) inflates in the outer bins — a resolution limit, not a bug
    b = rng.uniform(-1, 1, size=n)
    g = rng.normal(size=n)
    cases = {
        "A=B^2 (A|B)": (b ** 2, b),          # A fully determined by B ...
        "A=B^2 (B|A)": (b, b ** 2),          # ... but B NOT recoverable from A (sign lost)
        "monotone exp": (np.exp(b), b),
        "independent": (rng.normal(size=n), b),
        "B=A+noise (A|B)": (g, g + rng.normal(size=n)),   # expect ~0.5
    }
    res = {}
    print(f"{'case':18s} {'eta2':>7s} {'eps2':>7s}")
    for name, (a, c) in cases.items():
        cb = quantile_bins(c, K)
        e, ep = eta2_eps2(a, cb, K)
        res[name] = (e, ep)
        print(f"{name:18s} {e:7.3f} {ep:7.3f}")
    checks = [
        ("A=B^2: eta2(A|B) > 0.95", res["A=B^2 (A|B)"][0] > 0.95),
        ("A=B^2: eta2(B|A) < 0.05 (asymmetry)", res["A=B^2 (B|A)"][0] < 0.05),
        ("monotone: eta2 > 0.95", res["monotone exp"][0] > 0.95),
        ("independent: |eps2| < 0.01", abs(res["independent"][1]) < 0.01),
        ("B=A+noise: eta2(A|B) in [0.4,0.6]", 0.4 < res["B=A+noise (A|B)"][0] < 0.6),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def patient_rows(patient_id: str):
    vols_t, label_t = load_patient(patient_id, "cpu")
    vols = {k: v.numpy().astype(np.float64) for k, v in vols_t.items()}
    masks = {k: m.numpy() for k, m in region_masks(label_t, vols_t["t1n"]).items()}
    variants = {"raw": vols,
                "highpass": {k: v - gaussian_filter(v, HIGHPASS_SIGMA) for k, v in vols.items()}}
    rows = []
    for region, mask in masks.items():
        n = int(mask.sum())
        for variant, imgs in variants.items():
            vals = {k: imgs[k][mask] for k in CONTRAST_SUFFIX}
            for K in KS:
                if n < MIN_VOX_PER_BIN * K:
                    continue
                bins = {k: quantile_bins(v, K) for k, v in vals.items()}
                for a, b in itertools.permutations(CONTRAST_SUFFIX, 2):
                    e, ep = eta2_eps2(vals[a], bins[b], K)
                    rows.append(dict(patient=patient_id, region=region, variant=variant, K=K,
                                     pred=a, cond=b, n_vox=n, eta2=e, eps2=ep))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()
    if args.sanity:
        sys.exit(run_sanity())

    patients = sorted(pd.read_csv(DATA_DIR / "patient_region_deltas.csv")["case"].unique())
    if args.limit_patients:
        patients = patients[: args.limit_patients]
    mine = patients[args.rank::args.world_size]
    shard = DATA_DIR / f"correlation_ratio_shard{args.rank}.csv"
    done = set(pd.read_csv(shard)["patient"].unique()) if shard.exists() else set()
    todo = [q for q in mine if q not in done]
    log.info("rank %d/%d: %d assigned, %d done, %d to do", args.rank, args.world_size,
             len(mine), len(done), len(todo))
    header = not shard.exists()
    for i, pid in enumerate(todo):
        try:
            rows = patient_rows(pid)
        except (FileNotFoundError, ValueError) as e:
            log.warning("SKIP %s: %s", pid, e)
            continue
        pd.DataFrame(rows, columns=COLS).to_csv(shard, mode="a", header=header, index=False)
        header = False
        if (i + 1) % 5 == 0:
            log.info("  rank %d: %d/%d", args.rank, i + 1, len(todo))
    log.info("rank %d finished", args.rank)


if __name__ == "__main__":
    main()
