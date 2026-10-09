#!/usr/bin/env python3
"""
Overlay a small number of GROUPS of already-computed causal-ablation ladders
(each produced by ladder_ood_common.py's run_ladder(), which writes a
per-dataset ladder_series.json) on one comparison figure per training
direction -- reusing ladder_ood_common.py's own plotting primitive
(_write_per_contrast_png) rather than a bespoke matplotlib script, so the
comparison figure shares the paper's colour/shading/significance convention
by construction instead of re-deriving it.

Why groups, not one line per source file: run_ladder()'s own per-dataset
plot draws the OOD-pooled mean plus each individual OOD contrast as a thin
sub-curve -- for a dataset with a single OOD item that sub-curve is
identical to the pooled mean, so the plot only shows one visible line.
There is nothing missing there; the comparison worth drawing is across
GENERALIZATION REGIMES for the same training direction, e.g. for the ispy2
breast task:
  - "standard contrast": ispy2's own in-dataset held-out contrast pooled
    with duke-breast-mri's cross-dataset test on its single acquired
    (t1wce/DCE) contrast.
  - "pre-contrast": duke's native-T1w, no-Gd test (duke only -- ispy2 has
    no pre-contrast acquisition).
A group's curve is the simple (source-level, unweighted-by-case) mean of
its members' own OOD-pooled dice/hd95 series -- matching this project's
established macro-averaging convention (equal weight per contrast/source,
not per case; see aggregate_from_config.py's "all" column and
significance_from_config.py's macro_perm).

Fill-swap significance colouring reuses _sig_color exactly as run_ladder's
own plots do, but each group's p-value here is the CONSERVATIVE max (least
significant) p-value across its constituent sources' own already-computed
fill_swap_significance -- ladder_series.json only carries each source's own
aggregate p-value, not the raw per-case pairs needed to re-run a proper
pooled Wilcoxon test across sources, so this script does not attempt to
fabricate a re-pooled test statistic. A group is only drawn as significant
if EVERY constituent source is independently significant in the same
direction, matching how the pooled OOD mean's own significance is read.

Usage:
  .venv/bin/python ladder_cross_dataset_plot.py <out_dir> <file_stem> <title> \
      "<group label 1>=<path1>[,<path2>,...]" "<group label 2>=<path1>[,...]" ...

  Writes <out_dir>/ladder_<file_stem>_cross_dataset_per_contrast.png (via
  ladder_ood_common.py's own _write_per_contrast_png, unmodified -- the
  "_per_contrast" suffix is that shared function's naming, kept as-is
  rather than renaming around it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_ood_common import _write_per_contrast_png  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _group_series_and_sig(loaded: list[dict]) -> tuple[dict, dict]:
    """loaded: list of a group's own ladder_series.json dicts.
    Returns (series, sig) in the same shape run_ladder() produces, so
    _write_per_contrast_png can't tell this group apart from a real
    single-source ladder's own OOD-pooled curve."""
    series = {}
    sig = {}
    for metric in ("dice", "hd95"):
        arrays = [d[metric] for d in loaded]
        n_rungs = len(arrays[0])
        series[metric] = [sum(a[r] for a in arrays) / len(arrays) for r in range(n_rungs)]

        ps = [d.get("fill_swap_significance", {}).get(metric, {}).get("pooled_p", float("nan"))
              for d in loaded]
        finite = [p for p in ps if p == p]  # drop NaN
        pooled_p = max(finite) if finite else float("nan")
        n_cases = sum(d.get("fill_swap_significance", {}).get(metric, {}).get("n_cases", 0)
                      for d in loaded)
        sig[metric] = {"pooled_p": pooled_p, "n_cases": n_cases}
    return series, sig


def main() -> None:
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(1)
    out_dir = Path(sys.argv[1])
    file_stem = sys.argv[2]
    title = sys.argv[3]
    group_args = []
    for arg in sys.argv[4:]:
        label, _, paths_str = arg.partition("=")
        if not paths_str:
            raise SystemExit(f"bad group (need label=path1,path2,...): {arg!r}")
        group_args.append((label, [Path(p) for p in paths_str.split(",")]))

    out_dir.mkdir(parents=True, exist_ok=True)

    common_labels = None
    per_contrast = {"dice": {}, "hd95": {}}
    group_sig = {"dice": {}, "hd95": {}}
    for label, paths in group_args:
        loaded = []
        for p in paths:
            if not p.is_file():
                print(f"  skip source for {label!r}: {p} not found")
                continue
            loaded.append(_load(p))
        if not loaded:
            print(f"  group {label!r}: no valid sources, skipping")
            continue
        if common_labels is None:
            common_labels = loaded[0]["labels"]
        series, sig = _group_series_and_sig(loaded)
        for metric in ("dice", "hd95"):
            per_contrast[metric][label] = series[metric]
            group_sig[metric][label] = sig[metric]["pooled_p"]
        print(f"group {label!r} ({len(loaded)} source(s)): "
              f"fill-swap Dice p={group_sig['dice'][label]:.2g} (worst of sources)")

    if common_labels is None:
        raise SystemExit("no valid groups found")

    rungs = [(label, "", "") for label in common_labels]
    group_labels = list(per_contrast["dice"].keys())
    # Bold pooled line = grand mean across groups; its own significance is the
    # same conservative all-groups-agree rule as each individual group's.
    overall_series = {
        metric: [sum(per_contrast[metric][g][r] for g in group_labels) / len(group_labels)
                  for r in range(len(common_labels))]
        for metric in ("dice", "hd95")
    }
    overall_sig = {
        metric: {"pooled_p": max(group_sig[metric].values()), "per_contrast": group_sig[metric]}
        for metric in ("dice", "hd95")
    }

    _write_per_contrast_png(out_dir, f"{file_stem}_cross_dataset", title, rungs,
                            overall_series, per_contrast, plot_labels=group_labels,
                            sig=overall_sig)


if __name__ == "__main__":
    main()
