#!/usr/bin/env python3
"""Aggregate the chimera experiment: per contrast/method, mean kidney Dice on the
pasted US region vs the center-prior control. Reads
METRICS_ROOT/chaos_model/<contrast>/chimera/<cat>_<run>/fold*/chimera_metrics.csv.
Writes a markdown table next to them and prints it.

Usage: python 06_08_aggregate_chimera.py [--metrics_root <chaos_model dir>]
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_root", required=True, help="…/02_metrics/chaos_model")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.metrics_root)
    # contrast -> run -> label -> list of dice
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for csvf in root.glob("*/chimera/*/fold*/chimera_metrics.csv"):
        contrast = csvf.parents[3].name
        run = csvf.parents[1].name
        for row in csv.DictReader(open(csvf)):
            try: data[contrast][run][row["label"]].append(float(row["dice"]))
            except ValueError: pass
    if not data:
        sys.exit(f"No chimera_metrics.csv under {root}/*/chimera/")
    lines = ["# TRUSTED chimera experiment — kidney Dice on the pasted US region", "",
             "Dice % (cross-fold mean). `kidney` = model prediction ∩ ROI vs placed US GT. "
             "`center-prior` = matched-size ball at ROI centroid (control). The experiment only "
             "shows context helps if **kidney > center-prior** AND beats the US-alone numbers.", ""]
    for contrast in sorted(data):
        lines += [f"## {contrast}", "", "| method | kidney Dice | center-prior Dice | n |", "|---|---|---|---|"]
        for run in sorted(data[contrast]):
            k = np.array(data[contrast][run].get("kidney", []), float)
            c = np.array(data[contrast][run].get("kidney_centerprior", []), float)
            lines.append(f"| {run} | {np.nanmean(k)*100:.1f} | "
                         f"{np.nanmean(c)*100:.1f} | {np.isfinite(k).sum()} |")
        lines.append("")
    out = Path(a.out) if a.out else root / "chimera_experiment_summary.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines)); print(f"→ {out}")

if __name__ == "__main__":
    main()
