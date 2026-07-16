#!/usr/bin/env python3
"""
Cross-dataset COMBINED significance — the headline significance for the paper.

Motivation (why this exists alongside significance_from_config.py):
  Each dataset×modality is an individually UNDERPOWERED experiment (few held-out
  cases → wide error bars → per-dataset p-values look "uncertain"). But if our method
  were merely *equal* to a competitor, the sign of the per-dataset difference would
  scatter ~50/50 across independent datasets. When our method instead wins in EVERY
  dataset, that consistency is itself strong evidence that the per-dataset tests never
  quantify. This driver combines the per-dataset results into one headline claim.

Unit of analysis = one trained model (one dataset × training-modality) = one
independent experiment. For each experiment we reuse EXACTLY the per-case,
fold-0-2-capped, OOD (training-contrast-excluded), macro-over-contrasts effect from
significance_from_config.py (same loader, same resolution), so the combined test is
consistent with the per-dataset tables.

Per competitor family, across the K experiments, we report:
  * wins/losses on the OOD macro effect direction,
  * mean OOD macroΔ (equal weight per experiment),
  * SIGN TEST  — binomial p (H0: P(win)=0.5); uses direction only (most conservative,
    makes no independence-of-magnitude assumption),
  * STOUFFER  — combines the K per-experiment two-sided Wilcoxon p-values with their
    directions into one Z (one-sided toward "ours better"); uses magnitude, so a few
    large wins dominate a couple of near-ties. Assumes the K experiments are independent.
  * DATASET-CLUSTERED sign test — experiments sharing a dataset (e.g. chaos t1in/t2spir)
    are NOT fully independent; this collapses each dataset to a single vote (all its
    experiments must agree) and runs the sign test on datasets — the conservative,
    independence-safe view.

Meta-config YAML:
  title: "Cross-dataset combined significance — OURS (train050_val000)"
  ref_pattern: "auglabAug_v26_6_2_train050_val000"   # substring picking the ref run per config
  competitors:                                        # display-name: substring
    auglab_default: "auglab_default"
    synthseg_EM: "synthseg_EM"
    srcsm: "srcsm"
  metric: dice                 # dice (higher better) | hd95 (lower better)
  exclude_in_domain: true      # exclude each config's in_domain_contrast from the OOD effect
  output_md: "${PROJECT_ROOT}/paper/meta_significance.md"
  experiments:
    - {name: "chaos t1in",  dataset: chaos,  config: "datasets/chaos/.../chaos_t1in_03_results.yaml"}
    - ...

`${VAR}` in paths/metrics_dirs is expanded from the environment (set METRICS_ROOT per
experiment automatically from its `dataset`, plus PROJECT_ROOT).

Usage:
  python meta_significance.py <meta_config.yaml>
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest, norm, wilcoxon

# Reuse the exact per-case loader + run resolution from the per-dataset script so the
# combined test lines up with the per-dataset tables (same folds, same resolution).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import significance_from_config as sig  # noqa: E402


def build_sources(cfg, dataset, project_root):
    """Expand a result config's sources (or single metrics_dir) with METRICS_ROOT set."""
    os.environ["METRICS_ROOT"] = f"{project_root}/datasets/{dataset}/8_results_{dataset}/02_metrics"
    if "sources" in cfg:
        return [{"metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                 "column_prefix": s.get("column_prefix", ""),
                 "column_rename": s.get("column_rename", {})} for s in cfg["sources"]]
    md = Path(os.path.expandvars(cfg["metrics_dir"]))
    return [{"metrics_dir": md, "column_prefix": "", "column_rename": {}}]


def pick_run(runs, pattern):
    """Return the single run id containing `pattern`; None if absent, first if several."""
    hits = [r for r in runs if pattern in r]
    return hits[0] if hits else None


def experiment_effect(cfg, dataset, ref_pattern, comp_pattern, metric, exclude_in_domain,
                      project_root):
    """One experiment's OOD result vs one competitor.

    Returns dict(macro, pooled_mean, p_two, n_pairs, n_contrasts, win) or None if the
    competitor / ref is absent or there is no overlapping data.
    """
    runs = cfg.get("runs", [])
    ref = pick_run(runs, ref_pattern)
    comp = pick_run(runs, comp_pattern)
    if ref is None or comp is None:
        return None
    sources = build_sources(cfg, dataset, project_root)
    in_domain = cfg.get("in_domain_contrast") if exclude_in_domain else None
    A = sig.load_run_cases(sources, ref, metric)
    B = sig.load_run_cases(sources, comp, metric)
    cols = set(A) | set(B)
    ood = [c for c in cols if c != in_domain]
    per_contrast, pooled = [], []
    for c in ood:
        x, y = sig.paired(A, B, c)          # aligned over shared cases
        if not len(x):
            continue
        per_contrast.append(float(np.mean(x - y)))
        pooled.extend((x - y).tolist())
    if not pooled:
        return None
    higher_better = metric == "dice"
    scale = 100 if higher_better else 1
    macro = float(np.mean(per_contrast)) * scale
    ax = np.array(pooled)
    try:
        p_two = float(wilcoxon(ax, np.zeros_like(ax), zero_method="wilcox").pvalue)
    except ValueError:
        p_two = 1.0
    # "win" = ours better on the macro effect (dice: macro>0; hd95: macro<0)
    win = (macro > 0) if higher_better else (macro < 0)
    return {"ref": ref, "comp": comp, "macro": macro, "pooled_mean": float(np.mean(ax)) * scale,
            "p_two": p_two, "n_pairs": len(ax), "n_contrasts": len(per_contrast), "win": win}


def combine(rows):
    """Sign test + Stouffer over a competitor's per-experiment rows."""
    k = len(rows)
    wins = sum(r["win"] for r in rows)
    sign_p = binomtest(wins, k, 0.5, alternative="two-sided").pvalue if k else float("nan")
    # Stouffer one-sided toward "ours better", using each experiment's direction.
    z = []
    for r in rows:
        one = r["p_two"] / 2 if r["win"] else 1 - r["p_two"] / 2
        one = min(max(one, 1e-16), 1 - 1e-16)
        z.append(norm.isf(one))
    Z = sum(z) / np.sqrt(k) if k else float("nan")
    stouffer_p = float(norm.sf(Z)) if k else float("nan")
    return wins, k - wins, sign_p, stouffer_p, float(np.mean([r["macro"] for r in rows]))


def cluster_sign_test(rows_by_dataset):
    """Dataset-level sign test: each dataset votes only if all its experiments agree."""
    votes = []  # +1 ours, -1 comp, None mixed
    for ds, rows in rows_by_dataset.items():
        if all(r["win"] for r in rows):
            votes.append(1)
        elif all(not r["win"] for r in rows):
            votes.append(-1)
        else:
            votes.append(None)
    decisive = [v for v in votes if v is not None]
    w = sum(1 for v in decisive if v == 1)
    p = binomtest(w, len(decisive), 0.5, alternative="two-sided").pvalue if decisive else float("nan")
    return w, len(decisive), len(votes), p


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_config")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    cfg = yaml.safe_load(Path(args.meta_config).read_text())
    ref_pattern = cfg["ref_pattern"]
    competitors = cfg["competitors"]
    metric = cfg.get("metric", "dice")
    exclude_in_domain = cfg.get("exclude_in_domain", True)
    title = cfg.get("title", "Cross-dataset combined significance")

    # Load every experiment's result config once.
    experiments = []
    for e in cfg["experiments"]:
        cfg_path = Path(os.path.expandvars(e["config"]))
        if not cfg_path.exists():
            print(f"  skip {e['name']}: config not found {cfg_path}", file=sys.stderr)
            continue
        experiments.append((e["name"], e["dataset"], yaml.safe_load(cfg_path.read_text())))

    lines = [f"# {title}", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}  |  metric: {metric}  |  "
             f"ref: `{ref_pattern}`  |  in-domain excluded: {exclude_in_domain}", "",
             f"Experiments (one trained model each): {len(experiments)}. "
             "Effect = OOD cross-contrast **macroΔ** (equal weight per held-out contrast, "
             "ref − competitor). **sign test** uses win/loss direction only; **Stouffer** "
             "combines the per-experiment Wilcoxon p-values with direction (assumes "
             "independence); **dataset-clustered** collapses same-dataset experiments to one "
             "vote (independence-safe). ✅ = combined p < "
             f"{args.alpha}.", ""]

    for disp, patt in competitors.items():
        rows, rows_by_ds = [], defaultdict(list)
        for name, dataset, rcfg in experiments:
            r = experiment_effect(rcfg, dataset, ref_pattern, patt, metric,
                                  exclude_in_domain, project_root)
            if r is None:
                continue
            r["name"], r["dataset"] = name, dataset
            rows.append(r)
            rows_by_ds[dataset].append(r)
        if not rows:
            lines += [f"## vs {disp}", "", "_no experiments with this competitor_", ""]
            continue
        wins, losses, sign_p, stouffer_p, mean_macro = combine(rows)
        dw, dn, dtot, dsign_p = cluster_sign_test(rows_by_ds)
        star = "✅" if np.isfinite(stouffer_p) and stouffer_p < args.alpha else ""
        lines += [
            f"## vs {disp}", "",
            f"- Experiments: **{wins}/{len(rows)} favor ours** (losses {losses}); "
            f"mean OOD macroΔ **{mean_macro:+.2f}**",
            f"- **Sign test** (direction only): p = **{fmt_p(sign_p)}**",
            f"- **Stouffer combined** (one-sided, ours better): p = **{fmt_p(stouffer_p)}** {star}",
            f"- **Dataset-clustered sign test**: {dw}/{dn} datasets favor ours "
            f"({dtot} total; mixed excluded) → p = **{fmt_p(dsign_p)}**", "",
            "| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |",
            "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['name']} | {r['dataset']} | {r['n_contrasts']} | {r['macro']:+.2f} | "
                         f"{r['pooled_mean']:+.2f} | {fmt_p(r['p_two'])} | "
                         f"{'ours' if r['win'] else 'competitor'} |")
        lines.append("")

    out = "\n".join(lines)
    print(out)
    if cfg.get("output_md"):
        p = Path(os.path.expandvars(cfg["output_md"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out + "\n")
        print(f"\n[written] {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
