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
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
from eval_folds import EVAL_FOLD_INDICES  # noqa: E402 — single source of truth, see eval_folds.py
from stat_tests import (  # noqa: E402 — shared math, see stat_tests.py
    wilcoxon_p, holm, fmt_p, macro_stat, macro_perm, macro_ci,
)

# Reuse run-dir resolution AND per-case loading from the aggregator so both
# scripts test the exact same numbers (aggregate_from_config.py now also draws
# an inline "sig. vs ref" column on the summary table using this same
# load_run_cases/paired — one implementation, two callers).
# aggregate_from_config.py is now a sibling in this same dir (00_03_evaluate/).
_AGG = Path(__file__).resolve().parent / "aggregate_from_config.py"
_spec = importlib.util.spec_from_file_location("aggregate_from_config", _AGG)
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)
resolve_run_dir = _agg.resolve_run_dir
load_run_cases = _agg.load_run_cases
load_run_cases_by_label = _agg.load_run_cases_by_label
paired = _agg.paired
# contrast_groups hierarchical pooling: implementation lives in
# aggregate_from_config.py (shared with its own inline sig column + "all"
# column, and with combined_modality_summary.py / meta_task_heatmap.py one
# level up) — see the block comment above resolve_group_value there.
resolve_group = _agg.resolve_group
_describe_group = _agg._describe_group


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
        raw_cols = [c for c in column_order if c in all_cols] + \
               [c for c in all_cols if c not in column_order]
    else:
        raw_cols = all_cols
    cols = raw_cols

    higher_better = args.metric == "dice"
    sign = 1.0 if higher_better else -1.0  # positive "diff" = ref is better

    scale = 100 if higher_better else 1
    unit = "Dice pts" if higher_better else "mm HD95"
    B_BOOT = 5000

    # contrast_groups (opt-in — see the block comment above resolve_group()). Disabled
    # by omitting the key entirely, or by `use_contrast_groups: false` (kept as an
    # explicit toggle so a config can carry a contrast_groups block for reference
    # without it being live — the "go back to current strategy" switch).
    contrast_groups = cfg.get("contrast_groups") if cfg.get("use_contrast_groups", True) else None
    group_note_lines = []

    if contrast_groups:
        label_data = {r: load_run_cases_by_label(sources, r, args.metric) for r in runs}
        group_names = list(contrast_groups.keys())
        # in_domain_contrast usually names a raw column (e.g. "open-ms_flair"), but a
        # group pooling that same modality across sources is typically named without
        # the dataset prefix (e.g. "flair"). Prefer an explicit in_domain_group; fall
        # back to in_domain_contrast only when it happens to also be a group name
        # (true for atlas-liver-hcc/chaos's configs, where the in-domain group was
        # deliberately kept same-named as its raw column).
        in_dom_group_cfg = cfg.get("in_domain_group", in_domain_contrast)
        in_dom = in_dom_group_cfg if in_dom_group_cfg in group_names else None
        ood_cols = [g for g in group_names if g != in_dom]
        ind_cols = [g for g in group_names if g == in_dom]
        cols = group_names  # drives the "ALL contrasts" block + its note text

        def diff_arrays(comp, subset):
            """List of per-group paired-diff arrays (ref − comp), each group
            resolved per its contrast_groups tree (may itself be a nested
            pool-then-average over organs/sources — see resolve_group)."""
            out = []
            for g in subset:
                arr = resolve_group(contrast_groups[g], label_data[ref], label_data[comp])
                if len(arr):
                    out.append(arr)
            return out

        group_note_lines = [
            "## Contrast-group pooling (active)", "",
            "Each headline \"contrast\" below is a **group**, not a raw column — pooled/averaged "
            "per this tree (list = pool cases across sources; nested dict = equal-weight average "
            "across the named children, e.g. organs, recursing to arbitrary depth). Raw per-column "
            "figures are still in the per-contrast transparency table at the bottom, unpooled.", "",
        ]
        for g in group_names:
            group_note_lines.append(f"- **{g}**:")
            group_note_lines += _describe_group(contrast_groups[g], indent=1)
        group_note_lines.append("")
    else:
        # In-domain (training) contrast. For multi-source configs the column is already
        # prefixed (e.g. "chaos_t2spir"); in_domain_contrast is written to match. OOD =
        # held-out contrasts that are NOT the training contrast; IND = training contrast only.
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

    def block_table(subset, heading, note):
        # HEADLINE = one-sided directional ("ours better") p, Holm-corrected across
        # competitors — justified because the hypothesis is directional (our method is
        # designed to improve generalization). Two-sided p kept as a secondary column;
        # note it, not the headline star, is what reveals a significant in-domain LOSS
        # (there the one-sided "ours better" p is ~1 and correctly unstarred).
        p1s, rows = [], []
        for comp in competitors:
            arrs = diff_arrays(comp, subset)
            obs, p2, p1 = macro_perm(arrs, higher_better, scale)
            lo, hi = macro_ci(arrs, scale, b_boot=B_BOOT, seed=0)
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
             "held-out cases where ref wins/loses (for a grouped contrast, counts pooled/organ-mean "
             "units, not raw cases — see the group tree below). Per-case scores are fold-0-2-capped "
             "means over labels.",
             "",
             f"Contrast-group pooling: **{'ON' if contrast_groups else 'OFF (flat per-column, current default strategy)'}**.",
             ""]
    lines += group_note_lines

    # HEADLINE — OOD cross-contrast generalization (training contrast EXCLUDED).
    lines += block_table(
        ood_cols,
        ("OOD cross-contrast generalization — training contrast EXCLUDED (headline)"
         if in_dom else "All held-out contrasts (no in-domain contrast declared)"),
        f"Held-out contrasts: {', '.join(ood_cols) if ood_cols else '(none)'}. "
        "This is the paper's headline claim (cross-contrast generalization).")

    # ALL contrasts combined (in-domain + OOD), equal weight per contrast — matches
    # the summary table's `all` column exactly (unlike OOD/IND, which deliberately
    # split that column apart for the headline claim).
    if in_dom:
        lines += block_table(
            cols, "ALL contrasts combined (in-domain + OOD)",
            f"All {len(cols)} contrasts: {', '.join(cols)}. Equal weight per contrast — "
            "the same estimand as the summary table's `all` column, just paired/tested "
            "rather than only averaged.")

    # IND — the training contrast only (shows the domain-randomization trade-off).
    if ind_cols:
        lines += block_table(
            ind_cols, "IND in-domain — training contrast only",
            f"In-domain contrast: {', '.join(ind_cols)}. Domain randomization can trade a little "
            "in-domain accuracy for OOD robustness, so macroΔ < 0 here vs a light-aug baseline "
            "(auglab_default) is an expected trade-off, not a failure.")

    # Per-contrast transparency: mean Δ + raw (uncorrected) Wilcoxon p, in-domain flagged.
    # Always the flat RAW columns (never grouped) — this is the ungrouped, un-pooled
    # ground truth the group tree above was built from; kept visible regardless of
    # whether contrast_groups is active.
    in_dom_raw = in_domain_contrast if (in_domain_contrast in raw_cols) else None
    lines += ["## Per-contrast mean Δ (ref − competitor) with raw Wilcoxon p", "",
              "_Raw, un-pooled columns — independent of contrast-group pooling above._", "",
              "| competitor | " + " | ".join((f"{c} *" if c == in_dom_raw else c) for c in raw_cols) + " |",
              "|" + "---|" * (len(raw_cols) + 1)]
    for comp in competitors:
        cells = []
        for col in raw_cols:
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
