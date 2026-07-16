#!/usr/bin/env python3
"""
Config-driven paired significance testing for cross-experiment segmentation results.

Common and shared: statistical primitives (wilcoxon_p, holm) live in the shared
datasets/00_commun_scripts/00_00_utils/stat_tests.py, imported here rather than
reimplemented — every dataset/experiment that wires this up gets the same math.

Companion to aggregate_from_config.py: consumes the SAME YAML configs (same
`sources`/`runs`/`column_order` fields, same run-ID resolution) so the tests line
up exactly with the summary tables. Used by chaos/on-harmony/open-ms (the
datasets on the YAML-config aggregation path); the amos/brats2024-glioma/
sliver07/trusted family uses the older CLI-args `aggregate_results.py`, which
doesn't have a YAML config for this script to consume.

Statistical model (why this is the right test):
  * Unit of analysis = the test CASE, not the fold. A case's score = mean Dice over its
    labels and over the evaluated folds → ONE score per (run, contrast, case). Treating
    each fold's re-scoring of the same patient as a separate sample is pseudo-replication.
  * Folds capped to EVAL_FOLD_INDICES (0-2 — see eval_folds.py), shared with the aggregator.
  * ESTIMAND = macroΔ: the mean OVER CONTRASTS of each contrast's mean paired difference
    (ref − competitor), i.e. EQUAL WEIGHT PER CONTRAST. This is exactly what the summary
    table's `all` column reports, so the test and the table finally agree. The earlier
    per-case *pooled* Wilcoxon weighted by case count (micro), which let one large external
    set dominate (e.g. chaos t2spir: the n=20 CT set drowned the n=4 MR-contrast wins →
    p=0.83 despite macroΔ=+1.3) — a real mismatch that misrepresented the truth.
  * TEST = contrast-stratified paired SIGN-FLIP permutation on macroΔ. Under H0 (no method
    difference) paired diffs are symmetric about 0, so flipping each case's sign is exact;
    no normality assumption, correct for small/skewed samples. 95% CI = hierarchical
    bootstrap (resample cases within each contrast). Holm-corrected across competitors.

Report sections per (reference vs each competitor):
  * HEADLINE — OOD cross-contrast generalization: TRAINING (in-domain) contrast EXCLUDED
    (per "cross-contrast generalization IS the headline"). macroΔ + bootstrap 95% CI +
    ONE-SIDED directional ("ours > comp") sign-flip p, Holm-corrected (headline, since the
    hypothesis is directional); two-sided p kept as a secondary column.
  * IND — the in-domain (training) contrast only: shows the domain-randomization trade-off
    (macroΔ < 0 vs a light-aug baseline like auglab_default is expected, not a failure).
  * per-contrast mean Δ + raw (uncorrected) Wilcoxon p, in-domain contrast flagged.
Requires `in_domain_contrast` in the config to split IND vs OOD; without it the headline
falls back to all contrasts (and says so).

NOTE: per-dataset tests are individually underpowered (few held-out cases); the CROSS-
DATASET combined test (sign test + Stouffer over all datasets) is the paper's strongest
statement — see datasets/00_commun_scripts/00_03_evaluate/meta_significance.py.

The reference ("Ours") run is auto-selected as the run whose key contains both
"auglabAug" and "v26_6_2" (the headline model) — CAUTION: ablation variants that
extend that same run id (e.g. "..._auglabAug_v26_6_2_noisefill_...") also match
this substring test and will be picked if they sort earlier in `runs:`. Pass
--ref <exact run id> explicitly whenever the run list contains such variants.

Usage:
  python significance_from_config.py <config.yaml> [--ref <substr>] [--metric dice]

Writes {output_dir}/{output_prefix}_significance.md and prints a summary.
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
from eval_folds import EVAL_FOLD_INDICES  # noqa: E402 — single source of truth, see eval_folds.py
from stat_tests import wilcoxon_p, holm, fmt_p  # noqa: E402 — shared math, see stat_tests.py

# Reuse run-dir resolution from the aggregator so resolution stays identical.
# aggregate_from_config.py is now a sibling in this same dir (00_03_evaluate/).
_AGG = Path(__file__).resolve().parent / "aggregate_from_config.py"
_spec = importlib.util.spec_from_file_location("aggregate_from_config", _AGG)
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)
resolve_run_dir = _agg.resolve_run_dir


def load_run_cases(sources, key, metric):
    """col -> {case_id: mean metric over that case's labels and folds in EVAL_FOLD_INDICES}."""
    per = defaultdict(lambda: defaultdict(list))
    for src in sources:
        run_dir = resolve_run_dir(src["metrics_dir"], key)
        if run_dir is None:
            continue
        prefix = src.get("column_prefix", "")
        rename = src.get("column_rename", {})
        for fold_dir in sorted(run_dir.glob("fold*")):
            try:
                fold_idx = int(fold_dir.name.replace("fold", ""))
            except ValueError:
                continue
            if fold_idx not in EVAL_FOLD_INDICES:
                continue
            csv_path = fold_dir / "eval_all.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    col = rename.get(row["group"], f"{prefix}{row['group']}")
                    try:
                        v = float(row[metric])
                    except (KeyError, ValueError):
                        continue
                    if np.isfinite(v):
                        per[col][row["case"]].append(v)
    return {col: {c: float(np.mean(vs)) for c, vs in cases.items() if vs}
            for col, cases in per.items()}


def paired(ref_cases, comp_cases, col):
    """Aligned (x_ref, y_comp) arrays over shared cases for one contrast column."""
    r, c = ref_cases.get(col, {}), comp_cases.get(col, {})
    common = sorted(set(r) & set(c))
    return np.array([r[k] for k in common]), np.array([c[k] for k in common])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ref", default=None,
                    help="EXACT run id (not just a substring) to use as reference. Default: "
                         "auto-pick the first run containing both 'auglabAug' and 'v26_6_2' — "
                         "pass this explicitly whenever the run list also contains ablation "
                         "variants of that same run (e.g. '..._noisefill_...'), since they "
                         "match the same substring test.")
    ap.add_argument("--metric", default="dice", choices=["dice", "hd95"])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    import yaml
    cfg = yaml.safe_load(cfg_path.open())

    if "sources" in cfg:
        sources = [{"metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                    "column_prefix": s.get("column_prefix", ""),
                    "column_rename": s.get("column_rename", {})} for s in cfg["sources"]]
        out_dir = Path(os.path.expandvars(cfg["output_dir"])) if "output_dir" in cfg \
            else sources[0]["metrics_dir"]
    else:
        md = Path(os.path.expandvars(cfg["metrics_dir"]))
        sources = [{"metrics_dir": md, "column_prefix": "", "column_rename": {}}]
        out_dir = md

    runs = cfg.get("runs", [])
    column_order = cfg.get("column_order", None)
    in_domain_contrast = cfg.get("in_domain_contrast", None)
    prefix = cfg.get("output_prefix", "significance")
    title = cfg.get("title", "Results")

    # Pick reference run. --ref matches EXACTLY (not a substring): ablation variants
    # of the reference (e.g. "..._noisefill_...") share the "auglabAug"+"v26_6_2"
    # substring, so a fuzzy match here could silently pick the wrong run.
    if args.ref:
        ref = args.ref if args.ref in runs else None
    else:
        ref = next((r for r in runs if "auglabAug" in r and "v26_6_2" in r), None) \
            or next((r for r in runs if "v26_6_2" in r), None)
    if ref is None:
        sys.exit(f"Could not identify a reference run; pass --ref <exact run id from {runs}>.")
    competitors = [r for r in runs if r != ref]

    # Load per-case scores for every run (EVAL_FOLD_INDICES-capped — see load_run_cases).
    data = {r: load_run_cases(sources, r, args.metric) for r in runs}

    all_cols = sorted({c for d in data.values() for c in d})
    if column_order:
        cols = [c for c in column_order if c in all_cols] + \
               [c for c in all_cols if c not in column_order]
    else:
        cols = all_cols

    higher_better = args.metric == "dice"
    sign = 1.0 if higher_better else -1.0  # positive "diff" = ref is better

    scale = 100 if higher_better else 1
    unit = "Dice pts" if higher_better else "mm HD95"
    rng = np.random.default_rng(0)      # reproducible bootstrap CI
    B_BOOT = 5000

    # In-domain (training) contrast. For multi-source configs the column is already prefixed
    # (e.g. "chaos_t2spir"); in_domain_contrast is written to match. OOD = held-out contrasts
    # that are NOT the training contrast; IND = the training contrast only.
    in_dom = in_domain_contrast if (in_domain_contrast in cols) else None
    ood_cols = [c for c in cols if c != in_dom]
    ind_cols = [c for c in cols if c == in_dom]

    def diff_arrays(comp, subset):
        """List of per-contrast paired-diff arrays (ref − comp over shared cases)."""
        out = []
        for col in subset:
            x, y = paired(data[ref], data[comp], col)
            if len(x):
                out.append(x - y)
        return out

    def macro_stat(arrs):
        return float(np.mean([a.mean() for a in arrs])) if arrs else float("nan")

    def macro_perm(arrs):
        """Contrast-stratified paired SIGN-FLIP test on the macroΔ (analytic normal form).

        macroΔ = mean over contrasts of that contrast's mean paired diff (equal weight per
        contrast — the estimand the summary `all` column reports, so the test can't be
        hijacked by case-count imbalance). Under H0 (no method difference) each paired diff
        d flips sign with E[±d]=0, Var[±d]=d². The sign-flip null of macroΔ therefore has
        mean 0 and Var = (1/K²) Σ_k (1/n_k²) Σ_i d_ki², closed-form (no simulation, no
        Monte-Carlo floor). Z = macroΔ / sd → normal-approximation two-sided and one-sided
        (ours-better) p. Returns (macroΔ*scale, two-sided p, one-sided p)."""
        if not arrs:
            return float("nan"), float("nan"), float("nan")
        obs = macro_stat(arrs)
        K = len(arrs)
        var_T = sum((a ** 2).sum() / (a.size ** 2) for a in arrs) / (K ** 2)
        sd = float(np.sqrt(var_T))
        if sd == 0:
            p_two = 1.0 if obs == 0 else 0.0
            return obs * scale, p_two, (0.0 if (obs > 0) == higher_better and obs != 0 else 1.0)
        z = obs / sd
        p_two = float(2 * norm.sf(abs(z)))
        p_one = float(norm.sf(z) if higher_better else norm.cdf(z))
        return obs * scale, p_two, p_one

    def macro_ci(arrs):
        """Hierarchical bootstrap 95% CI of macroΔ (resample cases within each contrast)."""
        if not arrs:
            return float("nan"), float("nan")
        per = np.empty((B_BOOT, len(arrs)))
        for j, a in enumerate(arrs):
            idx = rng.integers(0, a.size, size=(B_BOOT, a.size))
            per[:, j] = a[idx].mean(axis=1)
        lo, hi = np.percentile(per.mean(axis=1), [2.5, 97.5])
        return lo * scale, hi * scale

    def block_table(subset, heading, note):
        # HEADLINE = one-sided directional ("ours better") p, Holm-corrected across
        # competitors — justified because the hypothesis is directional (our method is
        # designed to improve generalization). Two-sided p kept as a secondary column;
        # note it, not the headline star, is what reveals a significant in-domain LOSS
        # (there the one-sided "ours better" p is ~1 and correctly unstarred).
        p1s, rows = [], []
        for comp in competitors:
            arrs = diff_arrays(comp, subset)
            obs, p2, p1 = macro_perm(arrs)
            lo, hi = macro_ci(arrs)
            better = (lambda a: (a > 0).sum()) if higher_better else (lambda a: (a < 0).sum())
            worse = (lambda a: (a < 0).sum()) if higher_better else (lambda a: (a > 0).sum())
            nw = int(sum(better(a) for a in arrs)); nl = int(sum(worse(a) for a in arrs))
            rows.append((comp, len(arrs), obs, lo, hi, p1, p2, nw, nl))
            p1s.append(p1)
        hp = holm(p1s)
        out = [f"## {heading}", "", note, "",
               "| competitor | #contr | macroΔ | 95% CI | **p (1-sided, ours>comp)** | Holm | p (2-sided) | case W/L |",
               "|---|---|---|---|---|---|---|---|"]
        for (comp, nc, obs, lo, hi, p1, p2, nw, nl), h in zip(rows, hp):
            star = "**" if np.isfinite(h) and h < args.alpha else ""
            ci = f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) else "—"
            out.append(f"| {comp} | {nc} | {obs:+.2f} | {ci} | {fmt_p(p1)} | "
                       f"{star}{fmt_p(h)}{star} | {fmt_p(p2)} | {nw}/{nl} |")
        return out + [""]

    lines = [f"# {title} — paired significance ({args.metric})", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Reference (Ours): `{ref}`  |  Folds: {', '.join(str(i) for i in EVAL_FOLD_INDICES)}  |  "
             f"In-domain (training) contrast: `{in_domain_contrast or 'n/a'}`", "",
             "Per-dataset test = **contrast-stratified paired sign-flip test** (analytic normal "
             "approximation — deterministic, no Monte-Carlo floor) on the "
             f"**macroΔ** (mean over contrasts of each contrast's mean paired diff, ref − competitor, "
             f"{unit}; equal weight per contrast, matching the summary `all` column — so one large "
             "external test set can't dominate). 95% CI = hierarchical bootstrap (resample cases "
             "within contrast). HEADLINE = **`p (1-sided, ours>comp)`** (Holm across competitors) — "
             "the hypothesis is directional (our method is designed to improve generalization); "
             "`p (2-sided)` is kept as a secondary column and is what flags a significant in-domain "
             "LOSS (where the 1-sided 'ours better' p is ≈1 and correctly unstarred). `case W/L` = "
             "held-out cases where ref wins/loses. Per-case scores are fold-0-2-capped means over labels.", ""]

    # HEADLINE — OOD cross-contrast generalization (training contrast EXCLUDED).
    lines += block_table(
        ood_cols,
        ("OOD cross-contrast generalization — training contrast EXCLUDED (headline)"
         if in_dom else "All held-out contrasts (no in-domain contrast declared)"),
        f"Held-out contrasts: {', '.join(ood_cols) if ood_cols else '(none)'}. "
        "This is the paper's headline claim (cross-contrast generalization).")

    # IND — the training contrast only (shows the domain-randomization trade-off).
    if ind_cols:
        lines += block_table(
            ind_cols, "IND in-domain — training contrast only",
            f"In-domain contrast: {', '.join(ind_cols)}. Domain randomization can trade a little "
            "in-domain accuracy for OOD robustness, so macroΔ < 0 here vs a light-aug baseline "
            "(auglab_default) is an expected trade-off, not a failure.")

    # Per-contrast transparency: mean Δ + raw (uncorrected) Wilcoxon p, in-domain flagged.
    lines += ["## Per-contrast mean Δ (ref − competitor) with raw Wilcoxon p", "",
              "| competitor | " + " | ".join((f"{c} *" if c == in_dom else c) for c in cols) + " |",
              "|" + "---|" * (len(cols) + 1)]
    for comp in competitors:
        cells = []
        for col in cols:
            x, y = paired(data[ref], data[comp], col)
            cells.append(f"{np.mean(x - y) * scale:+.2f} ({fmt_p(wilcoxon_p(x, y))})" if len(x) else "—")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    lines += ["", "_`*` = in-domain (training) contrast._", "",
              "_Cross-dataset combined test (sign test + Stouffer over all datasets): "
              "`scripts/evaluate/run_meta_significance.sh`._"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_significance.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
