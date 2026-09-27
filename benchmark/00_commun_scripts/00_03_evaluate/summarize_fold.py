#!/usr/bin/env python3
"""
Per-fold evaluation summariser, shared across datasets.

Merges the per-group metric CSVs a fold produced (``<group>_metrics.csv``, written by
the common evaluator) into a single ``eval_all.csv``, and renders a human-readable
``eval_summary.md`` (group × label table of mean±std Dice / HD95).

This replaces the inline Python heredocs that were copy-pasted into every dataset's
06_01_evaluate_run.sh. ``eval_all.csv`` is the machine contract consumed by the
aggregators — it is byte-for-byte what the old heredocs wrote (same DictReader→DictWriter
over the same fieldnames). ``eval_summary.md`` is human-only and now uses one uniform
layout across datasets (title/wording tuned via the flags below).

Usage:
  python summarize_fold.py <eval_dir> <run_id> <fold> --groups g1 [g2...]
      [--group-col modality] [--groups-word Modalities] [--label-word Labels]
      [--title-suffix " | AMOS CT+MRI | chaos-trained models"] [--note "..."]...
"""
import argparse
import csv
from pathlib import Path

import numpy as np

CSV_FIELDS = ["group", "case", "label", "dice", "hd95"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("run_id")
    ap.add_argument("fold")
    ap.add_argument("--groups", nargs="+", required=True,
                    help="group names whose <group>_metrics.csv to merge, in order")
    ap.add_argument("--group-col", default="modality",
                    help="singular column header for the group axis (e.g. modality, contrast)")
    ap.add_argument("--groups-word", default="Modalities",
                    help="plural word for the group axis in the metadata line")
    ap.add_argument("--label-word", default="Labels",
                    help="word for the label axis in the metadata line (e.g. Labels, Organs)")
    ap.add_argument("--title-suffix", default="",
                    help="appended to the '# Evaluation — <run> (fold <f>)' title")
    ap.add_argument("--note", action="append", default=[],
                    help="extra metadata line(s) under the title (repeatable)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)

    rows: list = []
    for g in args.groups:
        p = eval_dir / f"{g}_metrics.csv"
        if p.exists():
            with p.open() as f:
                rows.extend(list(csv.DictReader(f)))

    with (eval_dir / "eval_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)

    labels = sorted({r["label"] for r in rows})

    def agg(g, lab, key):
        v = np.array([float(r[key]) for r in rows
                      if r["group"] == g and r["label"] == lab], float)
        n = int(np.isfinite(v).sum())
        return (np.nanmean(v) if n else float("nan"),
                np.nanstd(v) if n else float("nan"), n)

    lines = [f"# Evaluation — {args.run_id} (fold {args.fold}){args.title_suffix}", ""]
    lines += list(args.note)
    lines += [f"{args.groups_word}: {', '.join(args.groups)}  |  "
              f"{args.label_word}: {', '.join(labels)}", ""]
    for metric, fmt in (("dice", "{:.4f}±{:.4f}"), ("hd95", "{:.2f}±{:.2f}")):
        title = "Dice (↑)" if metric == "dice" else "HD95 mm (↓)"
        lines += [f"## {title}", "",
                  f"| {args.group_col} | " + " | ".join(labels) + " |",
                  "|" + "---|" * (len(labels) + 1)]
        for g in args.groups:
            cells = []
            for lab in labels:
                m, s, n = agg(g, lab, metric)
                cells.append("—" if not np.isfinite(m) else fmt.format(m, s))
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")

    (eval_dir / "eval_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"→ {eval_dir}/eval_summary.md")


if __name__ == "__main__":
    main()
