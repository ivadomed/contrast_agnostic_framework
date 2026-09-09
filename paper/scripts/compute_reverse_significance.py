#!/usr/bin/env python3
"""
For every method that OUT-performs OURS on a results table, is that difference
actually SIGNIFICANT?

Motivation: the `sig. vs ref` column in every generated table (see
aggregate_from_config.significance_column) is a ONE-SIDED test of "ref (OURS) is
better". A large p there means only "we cannot show OURS is better" -- it does
NOT establish that the competitor beats us. Those are different questions, and
the distinction matters exactly where OURS does not win: on the mandibular
(ToothFairy2) and breast (I-SPY2/Duke) tasks, a competitor leads on the point
estimate, and the paper needs to say whether that lead is real or noise.

This script reports, per competitor:
  macroD      OURS minus competitor on the `all` estimand (negative = they lead)
  p_ours      one-sided "OURS better"      <- what the table already prints
  p_them      one-sided "competitor better"  <- the missing direction
  p_two       two-sided "they differ at all"
each Holm-corrected within its own direction's family.

No new statistical machinery: the same load_run_cases/paired/macro_perm/holm
used by the shared aggregate layer, with the paired differences negated to swap
the tested direction (arrs are ref-minus-competitor, so -arrs is
competitor-minus-ref and macro_perm's "ref better" becomes "competitor better").

Usage:
  .venv/bin/python compute_reverse_significance.py <config.yaml> [<config.yaml> ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_03_evaluate"))
from aggregate_from_config import (  # noqa: E402
    load_run_cases, load_run_cases_by_label, paired, resolve_group,
    find_ref_key, load_run_from_sources,
)
from stat_tests import holm, macro_perm, fmt_p  # noqa: E402


def _expand(raw: str) -> Path:
    """Expand a config path, refusing to silently proceed on an UNSET variable.

    These configs reference dataset-scoped vars like ${METRICS_ROOT} that are
    set by the dataset's 00_utils/env.sh. os.path.expandvars leaves an unset
    var as the literal text '${METRICS_ROOT}', which yields a path that simply
    does not exist -- and elsewhere in this project that exact failure silently
    CREATED a directory literally named '${METRICS_ROOT}' full of misfiled
    results. Fail loudly instead: source the dataset's env.sh first."""
    out = os.path.expandvars(raw)
    if "${" in out or (out.startswith("$") and "/" in out):
        sys.exit(f"ERROR: unexpanded variable in config path: {out!r}\n"
                 f"       Source the dataset's 00_utils/env.sh first, e.g.\n"
                 f"       set -a; source datasets/<ds>/5_scripts_<ds>/00_utils/env.sh; set +a")
    return Path(out)


def _sources_from_cfg(cfg):
    if "sources" in cfg:
        return [
            {
                "metrics_dir": _expand(s["metrics_dir"]),
                "column_prefix": s.get("column_prefix", ""),
                "column_rename": s.get("column_rename", {}),
            }
            for s in cfg["sources"]
        ]
    return [{"metrics_dir": _expand(cfg["metrics_dir"]),
             "column_prefix": "", "column_rename": {}}]


def _diff_arrays(sources, ref_key, comp_key, metric, all_contrasts, contrast_groups):
    """Per-contrast paired (ref - competitor) arrays -- exactly the `arrs` that
    significance_column feeds to macro_perm."""
    if contrast_groups:
        ref_by_label = load_run_cases_by_label(sources, ref_key, metric)
        comp_by_label = load_run_cases_by_label(sources, comp_key, metric)
        arrs = []
        for g in contrast_groups.values():
            arr = resolve_group(g, ref_by_label, comp_by_label)
            if len(arr):
                arrs.append(arr)
        return arrs
    ref_cases = load_run_cases(sources, ref_key, metric)
    comp_cases = load_run_cases(sources, comp_key, metric)
    arrs = []
    for col in all_contrasts:
        x, y = paired(ref_cases, comp_cases, col)
        if len(x):
            arrs.append(x - y)
    return arrs


def analyse(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    sources = _sources_from_cfg(cfg)
    run_keys = cfg.get("runs", [])
    contrast_groups = (cfg.get("contrast_groups")
                       if cfg.get("use_contrast_groups", True) else None)
    ref_key = cfg.get("sig_ref") or find_ref_key(run_keys)
    if ref_key is None:
        print(f"  (no OURS run in {config_path.name} -- skipping)")
        return

    runs_data = {k: load_run_from_sources(sources, k)[0] for k in run_keys}
    all_contrasts = sorted({c for d in runs_data.values() for c in d.get("dice", {})})
    competitors = [k for k in run_keys if k != ref_key]

    print(f"\n{'=' * 78}\n{cfg.get('title', config_path.name)}\n  config: {config_path}")
    print(f"  ref (OURS): {ref_key}")

    for metric in ("dice", "hd95"):
        higher_better = metric == "dice"
        rows, p_ours_raw, p_them_raw, p_two_raw = [], [], [], []
        for key in competitors:
            arrs = _diff_arrays(sources, ref_key, key, metric, all_contrasts, contrast_groups)
            if not arrs:
                continue
            # scale: Dice is stored 0-1, so x100 to report Dice POINTS (matching
            # the summary tables); HD95 is already mm.
            scale = 100.0 if metric == "dice" else 1.0
            macro_d, p_two, p_ours = macro_perm(arrs, higher_better, scale=scale)
            # Same test, opposite direction: negate the paired diffs so
            # "ref better" becomes "competitor better".
            _, _, p_them = macro_perm([-a for a in arrs], higher_better)
            rows.append((key, macro_d))
            p_ours_raw.append(p_ours)
            p_them_raw.append(p_them)
            p_two_raw.append(p_two)
        if not rows:
            continue
        p_ours_adj, p_them_adj, p_two_adj = holm(p_ours_raw), holm(p_them_raw), holm(p_two_raw)

        unit = "Dice pts" if metric == "dice" else "mm"
        better = "higher" if higher_better else "lower"
        print(f"\n  --- {metric.upper()} ({better} is better) ---")
        print(f"  {'method':<52} {'macroD':>9} {'p_ours':>9} {'p_them':>9} {'p_two':>9}")
        for (key, macro_d), po, pt, p2 in zip(rows, p_ours_adj, p_them_adj, p_two_adj):
            # macroD is always ref-minus-competitor; "they lead" depends on metric direction.
            they_lead = (macro_d < 0) if higher_better else (macro_d > 0)
            flag = ""
            if they_lead:
                flag = "  <- THEY LEAD, SIGNIFICANTLY" if (np.isfinite(pt) and pt < 0.05) \
                    else "  <- they lead, NOT significant"
            print(f"  {key[:52]:<52} {macro_d:+9.2f} {fmt_p(po):>9} {fmt_p(pt):>9} {fmt_p(p2):>9}{flag}")
        print(f"  (macroD = OURS minus competitor, in {unit}; "
              f"p_ours/p_them one-sided, p_two two-sided; each Holm-corrected within its family)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        analyse(Path(os.path.expandvars(arg)))
