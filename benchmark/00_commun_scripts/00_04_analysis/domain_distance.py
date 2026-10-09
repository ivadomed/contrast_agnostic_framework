#!/usr/bin/env python3
"""
Model-free domain-distance between a TRAINING cohort and a set of evaluation
contrasts.

Motivation (2026-09-01, atlas-liver-hcc): an eval contrast that a *baseline*
(non-contrast-agnostic) model handles unusually well is a hint that the contrast
may not actually be out-of-domain. Deciding domain membership from Dice would be
circular, so this script measures domain membership from the IMAGES ALONE -- no
model, no metric, no run ids -- and is validated in-place by checking that the
contrasts already known to be in-domain come out closest.

Two measures per test contrast, both computed on per-volume appearance features
(robust-normalised in-body intensity histogram + fat-shell/interior ratio, i.e.
contrast/appearance rather than resolution or geometry):

  domain-classifier AUC : cross-validated logistic regression, train cohort vs
                          this contrast. 0.5 = indistinguishable (in-domain),
                          1.0 = trivially separable (out-of-domain). This is the
                          headline number.
  energy distance       : two-sample distributional distance on the same
                          features (scale-free, no classifier fitting).

Reading the output: rank the contrasts. The measure is only trustworthy if the
KNOWN in-domain contrasts land at the bottom -- that internal validation is
printed alongside, and a run where it fails should not be interpreted.

Usage:
  python domain_distance.py --ref-name atlas --ref-glob '<dir>/im*.nii.gz' \
      --test t2wi='<dir>/*_T2w.nii.gz' --test dwi='<dir>/*_dwi.nii.gz' ... \
      --known-in-domain ce-art,ce-pre,ce-pv,ce-del \
      --out <output_dir> [--max-per-set 60] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

N_BINS = 32
N_SLICES = 5


def _body_mask(sl: np.ndarray) -> np.ndarray | None:
    if sl.max() <= 0:
        return None
    pos = sl[sl > 0]
    if pos.size < 100:
        return None
    body = ndi.binary_fill_holes(sl > max(np.percentile(pos, 30), sl.max() * 0.06))
    if body.sum() < 500:
        return None
    lbl, n = ndi.label(body)
    if n > 1:
        body = lbl == (np.argmax(np.bincount(lbl.ravel())[1:]) + 1)
    return body


def volume_features(path: Path) -> np.ndarray | None:
    """Per-volume appearance features: in-body intensity histogram (robust
    per-image normalisation, so absolute scanner scaling drops out) + the
    fat-shell/interior ratio that distinguishes fat-suppressed from
    fat-bright acquisitions."""
    try:
        d = nib.load(str(path)).get_fdata()
    except Exception:
        return None
    if d.ndim != 3 or d.shape[2] < 5:
        return None
    zc = d.shape[2] // 2
    zs = np.linspace(max(0, zc - d.shape[2] // 5),
                     min(d.shape[2] - 1, zc + d.shape[2] // 5), N_SLICES).astype(int)

    hists, ratios = [], []
    for z in zs:
        sl = d[:, :, z].astype(np.float32)
        body = _body_mask(sl)
        if body is None:
            continue
        vals = sl[body]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            continue
        norm = np.clip((vals - lo) / (hi - lo), 0, 1)
        h, _ = np.histogram(norm, bins=N_BINS, range=(0, 1), density=True)
        hists.append(h)

        er = ndi.binary_erosion(body, iterations=4)
        shell, interior = body & ~er, ndi.binary_erosion(er, iterations=6)
        if shell.sum() >= 50 and interior.sum() >= 200:
            ratios.append(np.median(sl[shell]) / max(np.median(sl[interior]), 1e-6))

    if not hists or not ratios:
        return None
    return np.concatenate([np.mean(hists, axis=0), [np.median(ratios)]])


def collect(pattern: str, max_n: int, seed: int) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        return np.empty((0, N_BINS + 1))
    if len(files) > max_n:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), max_n, replace=False))]
    feats = [f for f in (volume_features(Path(p)) for p in files) if f is not None]
    return np.array(feats) if feats else np.empty((0, N_BINS + 1))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    def md(x, y):
        return np.mean(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2))
    return float(2 * md(a, b) - md(a, a) - md(b, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-name", required=True)
    ap.add_argument("--ref-glob", required=True)
    ap.add_argument("--test", action="append", required=True,
                    help="NAME=GLOB, repeatable")
    ap.add_argument("--known-in-domain", default="",
                    help="comma-separated test names known a priori to be in-domain "
                         "(used only to validate the measure, never to fit it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-set", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ref] {args.ref_name}: {args.ref_glob}", flush=True)
    ref = collect(args.ref_glob, args.max_per_set, args.seed)
    print(f"[ref] {len(ref)} volumes", flush=True)
    if len(ref) < 10:
        sys.exit(f"too few reference volumes ({len(ref)})")

    results = {}
    for spec in args.test:
        name, _, pattern = spec.partition("=")
        feats = collect(pattern, args.max_per_set, args.seed)
        print(f"[test] {name}: {len(feats)} volumes", flush=True)
        if len(feats) < 10:
            print(f"[test] {name}: SKIPPED (too few)", flush=True)
            continue

        X = np.vstack([ref, feats])
        y = np.r_[np.zeros(len(ref)), np.ones(len(feats))]
        Xs = StandardScaler().fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        prob = cross_val_predict(
            LogisticRegression(max_iter=5000, C=1.0), Xs, y, cv=cv,
            method="predict_proba")[:, 1]
        results[name] = {
            "n_test": int(len(feats)),
            "auc": float(roc_auc_score(y, prob)),
            "energy_distance": energy_distance(ref, feats),
            "shell_ratio_median": float(np.median(feats[:, -1])),
        }

    ref_shell = float(np.median(ref[:, -1]))
    known = [k.strip() for k in args.known_in_domain.split(",") if k.strip()]
    ranked = sorted(results.items(), key=lambda kv: kv[1]["auc"])

    lines = [
        f"# Domain distance from training cohort `{args.ref_name}` (model-free)",
        "",
        f"Reference: {len(ref)} volumes, median fat-shell/interior ratio {ref_shell:.2f}.",
        "",
        "AUC = cross-validated domain-classifier separability on per-volume appearance "
        "features. **0.5 = indistinguishable from the training cohort (in-domain); "
        "1.0 = trivially separable (out-of-domain).** Computed from images only — no "
        "model, no Dice, no run ids.",
        "",
        "| contrast | n | domain-classifier AUC | energy dist. | fat-shell ratio | known IND |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in ranked:
        flag = "yes" if name in known else ""
        lines.append(f"| {name} | {r['n_test']} | {r['auc']:.3f} | "
                     f"{r['energy_distance']:.3f} | {r['shell_ratio_median']:.2f} | {flag} |")

    if known:
        kn = [r["auc"] for n, r in results.items() if n in known]
        un = [r["auc"] for n, r in results.items() if n not in known]
        lines += ["", "## Internal validation", "",
                  f"Known in-domain contrasts ({', '.join(known)}): mean AUC "
                  f"{np.mean(kn):.3f}. Others: mean AUC {np.mean(un):.3f}.",
                  "", "The measure is only interpretable if the known in-domain "
                  "contrasts rank lowest; check the table above before drawing "
                  "conclusions about any other contrast."]

    (out_dir / "domain_distance.md").write_text("\n".join(lines) + "\n")
    (out_dir / "domain_distance.json").write_text(
        json.dumps({"ref": args.ref_name, "ref_shell_ratio": ref_shell,
                    "n_ref": int(len(ref)), "results": results}, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir/'domain_distance.md'}")


if __name__ == "__main__":
    main()
