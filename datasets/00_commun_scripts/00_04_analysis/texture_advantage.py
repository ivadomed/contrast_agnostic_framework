#!/usr/bin/env python3
"""
Texture-advantage analysis — quantify WHERE an image-driven augmentation (v26+auglab)
beats a label-driven generative one (SynthSeg EM), and by how much, per structure and
per test-contrast.

Motivation (see also datasets/<ds> CLAUDE.md notes): SynthSeg renders each label region
as `mean + std*N(0,1)` (flat mean + white noise; EM only subdivides the mosaic), so it is
structurally TEXTURE-BLIND. v26+auglab instead applies an affine remap `y = mu + a*(x-mean)`
to the REAL image, preserving within-region texture / gradients / fine structure. The
prediction is that the v26 advantage concentrates on small/thin/texture-defined structures
and in the failure tail, and is near-zero (or negative) on large mean-separable structures
or extreme-modality shifts (US, BOLD) where real texture does not transfer.

This is a method-agnostic re-analysis of EXISTING per-case metric CSVs (the
`<exp>/fold{F}/<testmod>_metrics.csv` files written by the shared evaluator,
00_03_evaluate/evaluate.py). It computes, per (test-contrast, label):
  * mean Dice of each method (pooled across folds)
  * Delta mean = ours - contender
  * Delta P25  = 25th-percentile Dice gap (tail / typical-bad-case)
  * Delta fail = failure-rate gap (% cases with Dice < FAIL_THRESH); negative = ours
                 fails on fewer cases.

It writes `texture_advantage_per_label.csv` + `summary.md` to --out_dir and prints the
biggest wins/losses. Reusable across datasets via the regex args.

Usage:
  python texture_advantage.py --metrics_base DIR --ours_regex RE --contender_regex RE \
      --out_dir DIR [--contender_name synthseg_EM] [--fail_thresh 0.5] [--dataset_tag NAME]
"""
import argparse, csv, re
from pathlib import Path
from collections import defaultdict

import numpy as np

FAIL_THRESH_DEFAULT = 0.5


def _latest_exp(base: Path, pattern: str):
    cands = [d for d in base.iterdir() if d.is_dir() and re.search(pattern, d.name)
             and "lowdata" not in d.name and "toDelete" not in d.name]
    return sorted(cands)[-1] if cands else None


def _load_per_case(exp_dir: Path):
    """{(testmod, label): {'dice': [...], 'hd95': [...]}} pooled across folds."""
    out = defaultdict(lambda: {"dice": [], "hd95": []})
    for fold in range(4):
        fd = exp_dir / f"fold{fold}"
        if not fd.exists():
            continue
        for csvf in fd.glob("*_metrics.csv"):
            testmod = csvf.name[: -len("_metrics.csv")]
            for r in csv.DictReader(open(csvf)):
                key = (testmod, r["label"])
                for metric in ("dice", "hd95"):
                    v = r.get(metric, "")
                    if v and v not in ("", "nan"):
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        if fv == fv:
                            out[key][metric].append(fv)
    return out


def _stats(vals, fail_thresh):
    a = np.asarray(vals, float)
    return dict(n=a.size, mean=a.mean() * 100,
                p25=np.percentile(a, 25) * 100,
                fail=(a < fail_thresh).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_base", required=True,
                    help="dir holding <exp>/fold{F}/<testmod>_metrics.csv subdirs")
    ap.add_argument("--ours_regex", required=True, help="regex matching OUR experiment dir name")
    ap.add_argument("--contender_regex", required=True, help="regex matching the contender exp dir name")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--contender_name", default="contender")
    ap.add_argument("--fail_thresh", type=float, default=FAIL_THRESH_DEFAULT)
    ap.add_argument("--dataset_tag", default="")
    args = ap.parse_args()

    base = Path(args.metrics_base)
    oe = _latest_exp(base, args.ours_regex)
    se = _latest_exp(base, args.contender_regex)
    if not oe or not se:
        raise SystemExit(f"Could not resolve experiments under {base}\n"
                         f"  ours_regex={args.ours_regex} -> {oe}\n"
                         f"  contender_regex={args.contender_regex} -> {se}")
    od, sd = _load_per_case(oe), _load_per_case(se)

    rows = []
    for key in sorted(set(od) & set(sd)):
        if len(od[key]["dice"]) < 4 or len(sd[key]["dice"]) < 4:
            continue
        o = _stats(od[key]["dice"], args.fail_thresh)
        s = _stats(sd[key]["dice"], args.fail_thresh)
        # HD95 boundary gap: contender_median − ours_median (positive ⇒ ours sharper).
        # NaN when either side has no finite HD95 for this cell (label often empty).
        oh, sh = od[key]["hd95"], sd[key]["hd95"]
        d_hd95 = (float(np.median(sh)) - float(np.median(oh))) if (oh and sh) else float("nan")
        rows.append(dict(testmod=key[0], label=key[1], n=min(o["n"], s["n"]),
                         ours_mean=o["mean"], cont_mean=s["mean"],
                         d_mean=o["mean"] - s["mean"], d_p25=o["p25"] - s["p25"],
                         d_fail=o["fail"] - s["fail"], d_hd95=d_hd95))
    rows.sort(key=lambda r: -r["d_mean"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "texture_advantage_per_label.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["testmod", "label", "n", "ours_mean",
                                          "cont_mean", "d_mean", "d_p25", "d_fail", "d_hd95"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    tag = args.dataset_tag or base.name
    md = [f"# Texture-advantage: ours vs {args.contender_name} — {tag}", "",
          f"Ours: `{oe.name}`  vs  `{se.name}`", "",
          f"Δ = ours − {args.contender_name} (Dice %). Δfail<0 ⇒ ours fails on fewer cases "
          f"(failure = Dice<{args.fail_thresh}).", "",
          "## Largest WINS (Δmean Dice)", "",
          "| testmod | label | n | ours | cont | Δmean | ΔP25 | Δfail% | ΔHD95mm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows[:20]:
        md.append(f"| {r['testmod']} | {r['label']} | {r['n']} | {r['ours_mean']:.1f} | "
                  f"{r['cont_mean']:.1f} | {r['d_mean']:+.1f} | {r['d_p25']:+.1f} | {r['d_fail']:+.1f} | "
                  f"{r['d_hd95']:+.1f} |")
    md += ["", "## Largest LOSSES (where contender wins)", "",
           "| testmod | label | n | ours | cont | Δmean | ΔP25 | Δfail% | ΔHD95mm |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows[-10:]:
        md.append(f"| {r['testmod']} | {r['label']} | {r['n']} | {r['ours_mean']:.1f} | "
                  f"{r['cont_mean']:.1f} | {r['d_mean']:+.1f} | {r['d_p25']:+.1f} | {r['d_fail']:+.1f} | "
                  f"{r['d_hd95']:+.1f} |")
    overall = np.mean([r["d_mean"] for r in rows]) if rows else float("nan")
    md += ["", f"**Overall mean Δ across {len(rows)} (testmod,label) cells: {overall:+.2f} Dice**", ""]
    (out / "summary.md").write_text("\n".join(md))

    print(f"[{tag}] ours={oe.name}\n        cont={se.name}")
    print(f"  {len(rows)} cells, overall mean Δ={overall:+.2f} Dice")
    print(f"  top wins: " + ", ".join(f"{r['testmod']}/{r['label']} {r['d_mean']:+.1f}" for r in rows[:5]))
    print(f"  → {csv_path}")


if __name__ == "__main__":
    main()
