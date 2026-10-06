#!/usr/bin/env python3
"""Audit one new val000 real-fill run's metrics against the OLD (val100) rung-5 run's metrics for the
same test source: per fold, the scored case set of every group must be identical (same test cases), and
mean Dice per group is printed side by side for a sanity read. Exit 1 on any mismatch.

  audit.py <old_run_metrics_dir> <new_run_metrics_dir> [label]
  audit.py expect:<group>=<n>[,<group>=<n>...] <new_run_metrics_dir> [label]   # no old reference on this cluster
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(d: Path, k: int):
    p = d / f"fold{k}" / "eval_all.csv"
    if not p.is_file():
        return None
    cases, dice = defaultdict(set), defaultdict(list)
    for r in csv.DictReader(open(p)):
        g = r.get("group") or r.get("contrast") or r.get("modality")
        cases[g].add(r["case"])
        try:
            v = float(r["dice"])
        except (ValueError, KeyError):
            continue
        if v == v:                       # NaN = label absent in that case: excluded, as in the aggregate layer
            dice[g].append(v)
    return cases, dice


def main():
    new = Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else new.name
    expect = None
    if sys.argv[1].startswith("expect:"):
        expect = {g: int(c) for g, c in (x.split("=") for x in sys.argv[1][7:].split(","))}
    else:
        old = Path(sys.argv[1])
    bad = 0
    for k in (0, 1, 2):
        if expect is not None:
            n = load(new, k)
            if n is None:
                print(f"[audit] {label} fold{k}: NEW eval_all.csv missing ({new})"); bad = 1; continue
            for g, c in expect.items():
                got = len(n[0].get(g, ()))
                d = sum(n[1][g]) / len(n[1][g]) * 100 if n[1].get(g) else float("nan")
                bad |= got != c
                print(f"[audit] {label} fold{k} {g:>18}: cases {got:4d} / expected {c:4d} {'OK ' if got == c else 'MISMATCH'}  Dice {d:5.1f}")
            continue
        o, n = load(old, k), load(new, k)
        if o is None:
            print(f"[audit] {label} fold{k}: OLD reference missing ({old}) -- cannot compare"); bad = 1; continue
        if n is None:
            print(f"[audit] {label} fold{k}: NEW eval_all.csv missing ({new})"); bad = 1; continue
        for g in sorted(set(o[0]) | set(n[0])):
            oc, nc = o[0].get(g, set()), n[0].get(g, set())
            od = sum(o[1][g]) / len(o[1][g]) * 100 if o[1].get(g) else float("nan")
            nd = sum(n[1][g]) / len(n[1][g]) * 100 if n[1].get(g) else float("nan")
            ok = oc == nc
            bad |= not ok
            print(f"[audit] {label} fold{k} {g:>18}: cases new {len(nc):4d} / old {len(oc):4d} "
                  f"{'OK ' if ok else 'MISMATCH'}  Dice new {nd:5.1f} old(val100) {od:5.1f}")
    print(f"[audit] {label}: {'AUDIT OK' if not bad else 'AUDIT FAILED'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
