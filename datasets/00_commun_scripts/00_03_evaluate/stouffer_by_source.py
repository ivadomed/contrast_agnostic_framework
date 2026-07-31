#!/usr/bin/env python3
"""
Source-level Stouffer combination for ONE trained model's cross-dataset results.

Why this exists (complements significance_from_config.py, does NOT replace it):
  significance_from_config.py collapses every contrast-column (own test set + each
  external cohort's contrasts) into ONE macroΔ with equal weight per column, and tests
  it with a contrast-stratified sign-flip permutation. That gives the reportable effect
  size, but it has two properties worth probing:
    1. A column backed by 8 cases and one backed by 100 count equally in the estimate;
       case count only enters via the shared variance term.
    2. Its variance treats all columns as independent strata — yet contrast-columns
       WITHIN one source are the SAME patients imaged in different contrasts
       (mslesseg_flair / _t1w / _t2w = one 115-subject cohort), so those strata are
       correlated and the variance is likely UNDERESTIMATED (anti-conservative).

  This script regroups the columns by SOURCE DATASET (the `sources:` entries of the
  same YAML config) and treats each source as one independent evidence stream:
  disjoint patient cohorts, different scanners, fixed trained model. It reports each
  source's own effect + p, then combines the K source-level one-sided p-values with
  STOUFFER's method:  Z = Σ Z_k / √K,  Z_k = Φ⁻¹(1 − p_k),  combined p = 1 − Φ(Z).
  Magnitude-aware (unlike a sign test) with a clean known null, so a source whose own
  n makes its evidence sharper naturally carries more weight — without raw case count
  dominating a pooled test.

  A √n-WEIGHTED Stouffer (Z = Σ w_k Z_k / √Σ w_k², w_k = √n_k) is reported alongside,
  since making the sample-size influence explicit is the exact question that motivated
  this. If unweighted and weighted agree, case-count imbalance is not driving the call.

  PATIENT-LEVEL mode (--patient-unit, default on) averages each patient's paired diff
  ACROSS that source's OOD contrasts before testing, so the unit of analysis is the
  patient, not the (patient × contrast) cell. This is the principled handling of the
  within-source correlation described above; --no-patient-unit reproduces the
  contrast-stratified treatment for comparison.

Reuses significance_from_config.py's loaders/resolution verbatim (same folds, same run
resolution) so numbers line up with the per-dataset tables.

Usage:
  python stouffer_by_source.py <config.yaml> --ref <exact run id> [--metric dice|hd95]
                               [--no-patient-unit]
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import significance_from_config as sig  # noqa: E402 — shared loaders, see its docstring

CLIP = 1e-16


def source_columns(src, runs_keys, metric):
    """Columns produced by ONE source (exact mapping, no prefix parsing)."""
    cols = set()
    for key in runs_keys:
        cols |= set(sig.load_run_cases([src], key, metric).keys())
    return cols


def stouffer(p_ones, weights=None):
    """Combine one-sided p-values. Returns (Z_combined, p_combined)."""
    if not p_ones:
        return float("nan"), float("nan")
    z = np.array([norm.isf(min(max(p, CLIP), 1 - CLIP)) for p in p_ones])
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float)
    Z = float((w * z).sum() / np.sqrt((w ** 2).sum()))
    return Z, float(norm.sf(Z))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ref", required=True, help="exact reference run id")
    ap.add_argument("--metric", default="dice", choices=["dice", "hd95"])
    ap.add_argument("--patient-unit", dest="patient_unit", action="store_true", default=True,
                    help="average each patient's diff across a source's OOD contrasts (default)")
    ap.add_argument("--no-patient-unit", dest="patient_unit", action="store_false",
                    help="keep (patient x contrast) cells as separate strata")
    ap.add_argument("--out", default=None, help="optional path to write markdown")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if "sources" not in cfg:
        sys.exit("This script needs a multi-source config (`sources:`).")
    sources = [{"metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                "column_prefix": s.get("column_prefix", ""),
                "column_rename": s.get("column_rename", {})} for s in cfg["sources"]]

    runs = cfg.get("runs", [])
    ref = args.ref
    if ref not in runs:
        sys.exit(f"--ref {ref} not in config runs: {runs}")
    competitors = [r for r in runs if r != ref]
    in_domain = cfg.get("in_domain_contrast")
    higher_better = args.metric == "dice"
    scale = 100 if higher_better else 1
    unit = "Dice pts" if higher_better else "mm HD95"

    # Exact column -> source mapping, and a readable label per source.
    src_cols, src_labels = [], []
    for src in sources:
        cols = source_columns(src, runs, args.metric)
        cols = {c for c in cols if c != in_domain}          # OOD only
        if not cols:
            continue
        src_cols.append(cols)
        label = src["column_prefix"].rstrip("_") or src["metrics_dir"].parts[-4]
        src_labels.append(label)

    data = {r: sig.load_run_cases(sources, r, args.metric) for r in runs}

    def source_effect(comp, cols):
        """(macroΔ*scale, p_two, p_one, n_unit) for one source's OOD columns."""
        if args.patient_unit:
            # One value per patient: mean paired diff across this source's contrasts.
            per_case = defaultdict(list)
            for col in cols:
                r, c = data[ref].get(col, {}), data[comp].get(col, {})
                for k in set(r) & set(c):
                    per_case[k].append(r[k] - c[k])
            if not per_case:
                return (float("nan"),) * 3 + (0,)
            arrs = [np.array([float(np.mean(v)) for v in per_case.values()])]
            n_unit = len(per_case)
        else:
            arrs, n_unit = [], 0
            for col in cols:
                x, y = sig.paired(data[ref], data[comp], col)
                if len(x):
                    arrs.append(x - y)
                    n_unit += len(x)
            if not arrs:
                return (float("nan"),) * 3 + (0,)

        # Same sign-flip math as significance_from_config.macro_perm.
        obs = float(np.mean([a.mean() for a in arrs]))
        K = len(arrs)
        var_T = sum((a ** 2).sum() / (a.size ** 2) for a in arrs) / (K ** 2)
        sd = float(np.sqrt(var_T))
        if sd == 0:
            p_two = 1.0 if obs == 0 else 0.0
            p_one = 0.0 if ((obs > 0) == higher_better and obs != 0) else 1.0
        else:
            z = obs / sd
            p_two = float(2 * norm.sf(abs(z)))
            p_one = float(norm.sf(z) if higher_better else norm.cdf(z))
        return obs * scale, p_two, p_one, n_unit

    mode = "patient-level (diffs averaged across each source's contrasts)" \
        if args.patient_unit else "contrast-stratified cells"
    lines = [f"# Source-level Stouffer — {cfg.get('title', 'Results')} ({args.metric})", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Reference (Ours): `{ref}`  |  metric: {args.metric} ({unit})  |  "
             f"in-domain excluded: `{in_domain}`  |  unit of analysis: {mode}", "",
             f"Each of the **{len(src_cols)} source cohorts** ({', '.join(src_labels)}) is treated as one "
             "independent evidence stream (disjoint patients, different scanners, fixed model). "
             "Per-source effect = mean paired diff (ref − competitor). Stouffer combines the "
             "per-source ONE-SIDED p-values: `Z = Σ Z_k/√K`. Weighted variant uses `w_k = √n_k` so "
             "sample-size influence is explicit — if unweighted and weighted agree, case-count "
             "imbalance is not driving the conclusion. "
             + ("For hd95, NEGATIVE Δ = ours better." if not higher_better
                else "For dice, POSITIVE Δ = ours better."), ""]

    for comp in competitors:
        rows, p_ones, ws = [], [], []
        for label, cols in zip(src_labels, src_cols):
            obs, p2, p1, n = source_effect(comp, cols)
            if not np.isfinite(obs):
                continue
            rows.append((label, n, obs, p2, p1))
            p_ones.append(p1)
            ws.append(np.sqrt(max(n, 1)))
        if not rows:
            continue
        Z_u, p_u = stouffer(p_ones)
        Z_w, p_w = stouffer(p_ones, ws)
        mean_eff = float(np.mean([r[2] for r in rows]))
        better = [(r[2] > 0) if higher_better else (r[2] < 0) for r in rows]

        lines += [f"## vs {comp}", "",
                  f"- Mean per-source Δ (equal weight per cohort): **{mean_eff:+.2f} {unit}**",
                  f"- Cohorts favoring ours: **{sum(better)}/{len(rows)}**",
                  f"- **Stouffer (unweighted)**: Z = {Z_u:+.2f}, one-sided p = **{p_u:.3g}**",
                  f"- **Stouffer (√n-weighted)**: Z = {Z_w:+.2f}, one-sided p = **{p_w:.3g}**", "",
                  "| source cohort | n units | Δ | p (1-sided, ours better) | p (2-sided) | favors |",
                  "|---|---|---|---|---|---|"]
        for (label, n, obs, p2, p1), b in zip(rows, better):
            lines.append(f"| {label} | {n} | {obs:+.2f} | {p1:.3g} | {p2:.3g} | "
                         f"{'ours' if b else 'competitor'} |")
        lines.append("")

    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n")
        print(f"\n[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
